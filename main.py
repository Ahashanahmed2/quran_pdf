#!/usr/bin/env python3
"""
FastAPI + HTML Form with HF Folder Check + Resume Support + Memory Monitor
"""

import os
import io
import re
import json
import time
import hashlib
import asyncio
import shutil
import traceback
import gc
import tempfile
import threading
import urllib.request
import urllib.error
import ssl
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
from huggingface_hub import HfApi, CommitOperationAdd

# ============ মেমরি মনিটর ============
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def get_memory_usage():
    """বর্তমান মেমরি ব্যবহার MB তে"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            mem = process.memory_info().rss / (1024 * 1024)
            return round(mem, 1)
        except:
            pass
    return 0

def get_system_memory():
    """সিস্টেম মেমরি তথ্য"""
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            return {
                "total": round(mem.total / (1024 * 1024), 0),
                "available": round(mem.available / (1024 * 1024), 0),
                "percent": mem.percent,
                "used": round(mem.used / (1024 * 1024), 0)
            }
        except:
            pass
    return {"total": 0, "available": 0, "percent": 0, "used": 0}

# ============ কনফিগারেশন ============
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")

TEMP_DIR = Path("/tmp/tafsir_temp")
TEMP_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = Path("/tmp/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MIN_FREE_SPACE_MB = 500
MAX_JOB_RUNTIME = 12 * 3600
PDF_SLEEP_BETWEEN = 3
BATCH_SIZE = 20
# =====================================

# গ্লোবাল ভেরিয়েবল
running_tasks = {}
running_tasks_lock = threading.Lock()
task_controls = {}
task_controls_lock = threading.Lock()
shutdown_in_progress = False

# SSL কনটেক্সট
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

class ProcessRequest(BaseModel):
    book_name: str
    urls: list

# ============ Disk Space Check ============
def check_disk_space_sync():
    try:
        total, used, free = shutil.disk_usage("/tmp")
        free_mb = free // (1024 * 1024)
        if free_mb < MIN_FREE_SPACE_MB:
            raise RuntimeError(f"Low disk space: {free_mb} MB free")
        return True
    except Exception as e:
        print(f"[Disk Check] Warning: {e}", flush=True)
        return True

# ============ Checkpoint ============
def get_checkpoint_path(folder_key):
    safe_key = folder_key.replace('/', '_').replace('\\', '_')
    return CHECKPOINT_DIR / f"checkpoint_{safe_key}.json"

def save_checkpoint_sync(folder_key, checkpoint):
    try:
        checkpoint_path = get_checkpoint_path(folder_key)
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        os.replace(temp_path, checkpoint_path)
    except Exception as e:
        print(f"[Checkpoint] Error: {e}", flush=True)

def load_checkpoint_sync(folder_key):
    try:
        checkpoint_path = get_checkpoint_path(folder_key)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"processed": [], "current": None, "last_page": 0}

# ============ HF ফোল্ডার চেক ============
def check_hf_folder_exists(hf_path: str) -> bool:
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_files_info(
            repo_id=HF_DATASET,
            repo_type="dataset",
            path=hf_path
        )
        return len(list(files)) > 0
    except Exception:
        return False

def get_hf_folder_progress(hf_path: str) -> dict:
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_files_info(
            repo_id=HF_DATASET,
            repo_type="dataset",
            path=hf_path
        )

        pdf_folders = set()
        for file in files:
            parts = file.rfilename.split('/')
            if len(parts) >= 2:
                folder_name = parts[1] if parts[0] == hf_path else parts[0]
                if folder_name.isdigit():
                    pdf_folders.add(folder_name)

        return {
            "exists": True,
            "uploaded_pdfs": sorted(list(pdf_folders)),
            "total_uploaded": len(pdf_folders)
        }
    except Exception as e:
        print(f"[HF Check] Error: {e}", flush=True)
        return {"exists": False, "uploaded_pdfs": [], "total_uploaded": 0}

# ============ PDF ডাউনলোড ============
def download_pdf_stream(url):
    print(f"[DEBUG] Downloading: {url[:80]}...", flush=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Referer': 'https://archive.org/'
        }

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, context=ssl_context, timeout=300) as response:
            total_size = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                temp_file.write(chunk)
                total_size += len(chunk)

            print(f"[DEBUG] Downloaded: {total_size} bytes", flush=True)

        temp_file.close()
        return temp_file.name

    except Exception as e:
        print(f"[DEBUG] Download error: {e}", flush=True)
        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except:
            pass
        raise e

# ============ স্ট্রিমিং আপলোড ফাংশন ============
def upload_folder_streaming_chunks(local_folder: str, hf_path: str, batch_name: str):
    max_retries = 3
    
    for retry in range(max_retries):
        try:
            png_files = sorted(Path(local_folder).glob("*.png"))
            file_count = len(png_files)
            
            if file_count == 0:
                print(f"[ERROR] No PNG files found", flush=True)
                return False

            print(f"[INFO] Uploading {file_count} images (attempt {retry+1}/{max_retries})", flush=True)
            
            api = HfApi(token=HF_TOKEN)
            
            operations = []
            
            for idx, png_file in enumerate(png_files, 1):
                remote_path = f"{hf_path}/{png_file.name}"
                
                with open(png_file, 'rb') as f:
                    content = f.read()
                
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=remote_path,
                        path_or_fileobj=content
                    )
                )
                
                if idx % 10 == 0:
                    print(f"[INFO] Prepared {idx}/{file_count} files", flush=True)
            
            print(f"[INFO] Creating single commit with {len(operations)} files...", flush=True)
            
            api.create_commit(
                repo_id=HF_DATASET,
                repo_type="dataset",
                operations=operations,
                commit_message=f"📚 {batch_name} - {file_count} pages"
            )
            
            print(f"[INFO] ✅ Uploaded {file_count} images", flush=True)
            return True
            
        except Exception as e:
            print(f"[ERROR] Upload attempt {retry+1} failed: {e}", flush=True)
            if retry < max_retries - 1:
                wait_time = 10 * (retry + 1)
                print(f"[INFO] Waiting {wait_time}s before retry...", flush=True)
                time.sleep(wait_time)
            else:
                traceback.print_exc()
                return False
    
    return False

def is_task_cancelled(task_id):
    with task_controls_lock:
        return task_controls.get(task_id, {}).get("cancel", False)

def process_single_pdf_sync(pdf, clean_folder_name, folder_key, checkpoint, task_id, pdf_index, total_pdfs):
    sub_folder = str(pdf['number'])
    full_hf_path = f"{clean_folder_name}/{sub_folder}"
    print(f"[INFO] Processing: {pdf['name']} -> {full_hf_path}", flush=True)

    start_page = 0
    if checkpoint.get('current') == pdf['name']:
        start_page = checkpoint.get('last_page', 0) + 1
        if start_page > 0:
            print(f"[INFO] Resuming from page {start_page + 1}", flush=True)

    if not pdf.get('download_link'):
        return 0, False

    print(f"[INFO] Downloading: {pdf['name']}", flush=True)
    try:
        pdf_path = download_pdf_stream(pdf['download_link'])
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", flush=True)
        return 0, False

    total_pages_processed = 0

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        del doc
        gc.collect()
        
        print(f"[INFO] {pdf['name']}: {total_pages} pages", flush=True)

        with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["current_pdf"] = pdf['name']
                running_tasks[task_id]["current_pdf_index"] = pdf_index
                running_tasks[task_id]["current_pdf_total_pages"] = total_pages
                running_tasks[task_id]["current_page"] = start_page
                # 🔥 মেমরি ব্যবহার আপডেট
                running_tasks[task_id]["memory_usage"] = get_memory_usage()
                running_tasks[task_id]["system_memory"] = get_system_memory()

        batch_number = (start_page // BATCH_SIZE) + 1
        
        for batch_start in range(start_page, total_pages, BATCH_SIZE):
            if shutdown_in_progress or is_task_cancelled(task_id):
                print(f"[INFO] Task cancelled", flush=True)
                return total_pages_processed, True
            
            batch_end = min(batch_start + BATCH_SIZE, total_pages)
            
            temp_folder = TEMP_DIR / f"{task_id}_{pdf['number']}_batch{batch_number}"
            temp_folder.mkdir(exist_ok=True)
            
            print(f"[INFO] 📦 Batch {batch_number}: Pages {batch_start+1}-{batch_end} [Mem: {get_memory_usage()}MB]", flush=True)
            
            batch_doc = None
            try:
                batch_doc = fitz.open(pdf_path)
                
                for page_num in range(batch_start, batch_end):
                    page = batch_doc.load_page(page_num)
                    zoom = 4.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    img_name = f"page_{page_num+1:04d}.png"
                    tmp_path = temp_folder / img_name
                    pix.save(tmp_path)
                    
                    total_pages_processed += 1
                    
                    with running_tasks_lock:
                        if task_id in running_tasks:
                            running_tasks[task_id]["current_page"] = page_num + 1
                            running_tasks[task_id]["memory_usage"] = get_memory_usage()
                    
                    del page
                    del pix
                    
                    if (page_num - batch_start + 1) % 5 == 0:
                        gc.collect()
                
                batch_doc.close()
                del batch_doc
                gc.collect()
                
                print(f"[INFO] 📤 Uploading batch {batch_number}... [Mem: {get_memory_usage()}MB]", flush=True)
                
                upload_success = upload_folder_streaming_chunks(
                    str(temp_folder), 
                    full_hf_path, 
                    f"{pdf['name']}_batch{batch_number}"
                )
                
                if upload_success:
                    print(f"[INFO] ✅ Batch {batch_number} uploaded", flush=True)
                    
                    try:
                        checkpoint['current'] = pdf['name']
                        checkpoint['last_page'] = batch_end - 1
                        save_checkpoint_sync(folder_key, checkpoint)
                    except Exception as e:
                        print(f"[ERROR] Checkpoint save failed: {e}", flush=True)
                    
                    batch_number += 1
                else:
                    raise Exception(f"Batch {batch_number} upload failed")
                    
            finally:
                if batch_doc:
                    try:
                        batch_doc.close()
                    except:
                        pass
                    del batch_doc
                
                try:
                    if temp_folder.exists():
                        shutil.rmtree(temp_folder, ignore_errors=True)
                except:
                    pass
                
                gc.collect()
                gc.collect()
                
                # 🔥 মেমরি আপডেট
                with running_tasks_lock:
                    if task_id in running_tasks:
                        running_tasks[task_id]["memory_usage"] = get_memory_usage()
                        running_tasks[task_id]["system_memory"] = get_system_memory()
                
                if batch_end < total_pages:
                    wait_time = 5
                    print(f"[INFO] ⏸️ Pausing {wait_time}s... [Mem: {get_memory_usage()}MB]", flush=True)
                    time.sleep(wait_time)

        print(f"[INFO] ✅ Complete: {pdf['name']} - {total_pages_processed} pages", flush=True)
        checkpoint['current'] = None
        checkpoint['last_page'] = total_pages
        save_checkpoint_sync(folder_key, checkpoint)
        
        return total_pages_processed, False

    finally:
        try:
            os.unlink(pdf_path)
        except:
            pass
        gc.collect()

def process_pdf_urls_sync(pdf_urls: list, folder_name: str, task_id: str):
    print(f"[DEBUG] ========== PROCESS STARTED ==========", flush=True)
    print(f"[DEBUG] Folder: {folder_name}", flush=True)
    print(f"[DEBUG] PDF Count: {len(pdf_urls)}", flush=True)

    try:
        clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')

        print(f"[INFO] Checking HF for existing folder: {clean_folder_name}", flush=True)
        hf_progress = get_hf_folder_progress(clean_folder_name)

        if hf_progress["exists"] and hf_progress["total_uploaded"] > 0:
            print(f"[INFO] 📁 Folder already exists on HF", flush=True)

        checkpoint = load_checkpoint_sync(f"direct_{clean_folder_name}")
        local_processed = set(checkpoint.get('processed', []))

        hf_processed = set(hf_progress.get("uploaded_pdfs", []))
        all_processed = local_processed.union(hf_processed)
        print(f"[INFO] Total already processed: {len(all_processed)} PDFs", flush=True)

        processed = list(all_processed)

        with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["completed_pdfs"] = len(processed)
                running_tasks[task_id]["memory_usage"] = get_memory_usage()
                running_tasks[task_id]["system_memory"] = get_system_memory()

        for idx, url in enumerate(pdf_urls):
            if shutdown_in_progress or is_task_cancelled(task_id):
                break

            name_match = re.search(r'/(\d+)\.pdf', url)
            pdf_number = int(name_match.group(1)) if name_match else (idx + 1)
            pdf_name = f"{pdf_number}.pdf"

            if str(pdf_number) in hf_processed:
                print(f"[INFO] ⏭️ Skipping {pdf_name} (already on HF)", flush=True)
                continue

            if pdf_name in local_processed:
                print(f"[INFO] ⏭️ Skipping {pdf_name} (already processed locally)", flush=True)
                continue

            pdf = {'name': pdf_name, 'number': pdf_number, 'download_link': url}
            print(f"[INFO] [{idx+1}/{len(pdf_urls)}] Starting {pdf_name}", flush=True)

            try:
                pages_processed, cancelled = process_single_pdf_sync(
                    pdf, clean_folder_name, f"direct_{clean_folder_name}", checkpoint, task_id, idx+1, len(pdf_urls)
                )
                if cancelled:
                    break

                processed.append(pdf_name)
                checkpoint['processed'] = list(set(checkpoint.get('processed', []) + [pdf_name]))
                save_checkpoint_sync(f"direct_{clean_folder_name}", checkpoint)
                print(f"[INFO] ✅ Completed: {pdf_name} ({pages_processed} pages)", flush=True)

                with running_tasks_lock:
                    if task_id in running_tasks:
                        running_tasks[task_id]["completed_pdfs"] = len(processed)
                        running_tasks[task_id]["current_pdf"] = None
                        running_tasks[task_id]["current_page"] = 0
                        running_tasks[task_id]["memory_usage"] = get_memory_usage()

                if idx < len(pdf_urls) - 1:
                    time.sleep(PDF_SLEEP_BETWEEN)

            except Exception as e:
                print(f"[ERROR] Failed: {pdf_name} - {e}", flush=True)
                traceback.print_exc()
                continue

        new_processed = len(processed) - len(all_processed)
        if new_processed > 0:
            print(f"[INFO] 🎉 Completed {new_processed} new PDFs!", flush=True)
        else:
            print(f"[INFO] ✅ All PDFs were already processed!", flush=True)

        with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "completed"
                running_tasks[task_id]["completed_at"] = time.time()
                running_tasks[task_id]["completed_pdfs"] = len(processed)
                running_tasks[task_id]["memory_usage"] = get_memory_usage()
    except Exception as e:
        print(f"[DEBUG] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error"] = str(e)

# ============ HTML টেমপ্লেট ============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>তাফসীর PDF প্রসেসর</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 950px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #1a5f7a; text-align: center; margin-bottom: 10px; font-size: 2em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: bold; color: #333; }
        textarea {
            width: 100%; padding: 12px 15px; border: 2px solid #ddd;
            border-radius: 10px; font-size: 14px; min-height: 180px;
            font-family: monospace;
        }
        textarea:focus { outline: none; border-color: #1a5f7a; }
        .btn {
            background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%);
            color: white; border: none; padding: 15px 30px;
            border-radius: 10px; font-size: 16px; font-weight: bold;
            cursor: pointer; width: 100%; transition: transform 0.2s;
        }
        .btn:hover { transform: scale(1.02); }
        .btn-group { display: flex; gap: 10px; margin-bottom: 20px; }
        .btn-secondary { background: #6c757d; flex: 1; }
        .btn-primary { background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%); flex: 2; }
        .result {
            margin-top: 20px; padding: 15px; border-radius: 10px; display: none;
        }
        .result.success {
            background: #d4edda; border: 1px solid #c3e6cb;
            color: #155724; display: block;
        }
        .result.error {
            background: #f8d7da; border: 1px solid #f5c6cb;
            color: #721c24; display: block;
        }
        .info-box {
            background: #e7f3ff; border: 1px solid #b8daff;
            border-radius: 10px; padding: 15px; margin-bottom: 20px;
        }
        .quick-buttons { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }
        .quick-btn {
            background: #e9ecef; border: 1px solid #ced4da;
            padding: 8px 15px; border-radius: 5px; cursor: pointer; font-size: 12px;
        }
        .quick-btn:hover { background: #dee2e6; }
        .status-badge {
            display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: bold;
        }
        .status-running { background: #fff3cd; color: #856404; }
        .status-completed { background: #d4edda; color: #155724; }
        .progress-bar-container {
            background: #e9ecef; height: 25px; border-radius: 15px;
            margin: 8px 0; overflow: hidden;
        }
        .progress-bar {
            background: linear-gradient(135deg, #1a5f7a, #0d3b4c);
            height: 25px; border-radius: 15px; text-align: center;
            color: white; font-size: 13px; line-height: 25px;
            width: 0%;
        }
        .progress-bar-page {
            background: linear-gradient(135deg, #28a745, #20c997);
            height: 20px; border-radius: 10px; text-align: center;
            color: white; font-size: 11px; line-height: 20px;
            width: 0%;
        }
        .task-card {
            background: #f8f9fa; padding: 15px; margin-bottom: 15px;
            border-radius: 10px; border-left: 4px solid #1a5f7a;
        }
        .page-progress {
            margin-top: 10px; padding: 10px; background: white;
            border-radius: 8px; border: 1px solid #dee2e6;
        }
        .memory-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 8px 12px; border-radius: 8px;
            margin-top: 10px; font-size: 13px;
        }
        .memory-bar-container {
            background: rgba(255,255,255,0.3); height: 8px;
            border-radius: 4px; margin-top: 5px; overflow: hidden;
        }
        .memory-bar {
            background: #ffd700; height: 8px; border-radius: 4px;
            width: 0%; transition: width 0.3s;
        }
        .memory-warning { color: #ff9800; }
        .memory-critical { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 তাফসীর PDF প্রসেসর</h1>
        <p class="subtitle">PDF থেকে ইমেজ কনভার্ট করে Hugging Face Dataset-এ আপলোড করুন</p>
        
        <div class="info-box">
            <h3>📋 ব্যবহার নির্দেশিকা</h3>
            <ul>
                <li>প্রথম লাইনে বইয়ের নাম লিখুন</li>
                <li>তারপর Internet Archive আইটেম লিংক দিন</li>
                <li>অথবা সরাসরি PDF লিংক দিন (প্রতি লাইনে একটি)</li>
                <li>আগের ফোল্ডার থাকলে অটোমেটিক স্কিপ হবে</li>
            </ul>
        </div>

        <div class="quick-buttons">
            <span class="quick-btn" onclick="fillTafsir()">📖 তাফসীর (1 PDF)</span>
            <span class="quick-btn" onclick="fillTafsirFull()">📚 তাফসীর (22 PDF)</span>
            <span class="quick-btn" onclick="fillTafsirItem()">📁 তাফসীর (আইটেম)</span>
            <span class="quick-btn" onclick="clearForm()">🗑️ ক্লিয়ার</span>
        </div>

        <form id="processForm">
            <div class="form-group">
                <label for="inputData">📄 বইয়ের নাম ও PDF লিংকসমূহ:</label>
                <textarea id="inputData" name="inputData" placeholder="উদাহরণ:
তাফসীর ফী যিলালিল কোরআন(২২ খন্ড)
https://archive.org/details/20260415_20260415_0945"></textarea>
            </div>
            
            <div class="btn-group">
                <button type="button" class="btn btn-secondary" onclick="previewOnly()">👁️ প্রিভিউ</button>
                <button type="submit" class="btn btn-primary">🚀 প্রসেসিং শুরু করুন</button>
            </div>
        </form>

        <div id="result" class="result"></div>
        
        <div style="margin-top: 25px;">
            <h3>📊 চলমান টাস্ক 
                <span style="font-size: 14px; margin-left: 10px;">
                    <a href="#" onclick="loadTasks(); return false;" style="color: #1a5f7a;">🔄 রিফ্রেশ</a>
                </span>
            </h3>
            <div id="tasksList">লোড হচ্ছে...</div>
        </div>
    </div>

    <script>
        async function loadTasks() {
            try {
                const response = await fetch('/tasks');
                const tasks = await response.json();
                const tasksDiv = document.getElementById('tasksList');
                
                if (Object.keys(tasks).length === 0) {
                    tasksDiv.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">কোনো চলমান টাস্ক নেই</p>';
                } else {
                    let html = '';
                    for (const [id, task] of Object.entries(tasks)) {
                        const statusClass = task.status === 'running' ? 'status-running' : 'status-completed';
                        
                        html += `<div class="task-card">`;
                        html += `<div style="display: flex; justify-content: space-between;">
                            <strong>🆔 ${id}</strong>
                            <span class="status-badge ${statusClass}">${task.status}</span>
                        </div>`;
                        
                        html += `<div style="margin-top: 8px;">`;
                        html += `📁 <strong>${task.folder_name || 'Unknown'}</strong><br>`;
                        html += `🕐 শুরু: ${new Date(task.started_at * 1000).toLocaleString('bn-BD')}<br>`;
                        
                        // 🔥 মেমরি ইনফো
                        if (task.memory_usage) {
                            const memUsage = task.memory_usage;
                            const sysMem = task.system_memory || { total: 512, available: 0, percent: 0 };
                            const memPercent = sysMem.total > 0 ? (memUsage / sysMem.total) * 100 : 0;
                            let memClass = '';
                            if (memPercent > 80) memClass = 'memory-critical';
                            else if (memPercent > 60) memClass = 'memory-warning';
                            
                            html += `<div class="memory-info">`;
                            html += `💾 মেমরি: ${memUsage} MB`;
                            if (sysMem.total > 0) {
                                html += ` / ${sysMem.total} MB (${sysMem.percent}%)`;
                            }
                            html += `<div class="memory-bar-container">
                                <div class="memory-bar" style="width: ${memPercent}%;"></div>
                            </div>`;
                            if (sysMem.available > 0) {
                                html += `<small>✅ উপলব্ধ: ${sysMem.available} MB</small>`;
                            }
                            html += `</div>`;
                        }
                        
                        if (task.total_pdfs) {
                            const completed = task.completed_pdfs || 0;
                            const percent = task.total_pdfs > 0 ? Math.round(completed * 100 / task.total_pdfs) : 0;
                            
                            html += `<div style="margin-top: 10px;">`;
                            html += `📊 <strong>PDF অগ্রগতি:</strong> ${completed} / ${task.total_pdfs} সম্পন্ন<br>`;
                            html += `<div class="progress-bar-container">
                                <div class="progress-bar" style="width: ${percent}%;">${percent}%</div>
                            </div>`;
                            
                            if (task.current_pdf) {
                                const currentPage = task.current_page || 0;
                                const totalPages = task.current_pdf_total_pages || 0;
                                const pagePercent = totalPages > 0 ? Math.round(currentPage * 100 / totalPages) : 0;
                                
                                html += `<div class="page-progress">`;
                                html += `🔄 <strong>চলমান:</strong> ${task.current_pdf} (${task.current_pdf_index}/${task.total_pdfs})<br>`;
                                html += `📄 <strong>পৃষ্ঠা:</strong> ${currentPage} / ${totalPages}<br>`;
                                html += `<div class="progress-bar-container" style="height: 20px;">
                                    <div class="progress-bar-page" style="width: ${pagePercent}%;">${pagePercent}%</div>
                                </div>`;
                                html += `</div>`;
                            }
                            
                            if (completed > 0 && task.started_at) {
                                const elapsed = Date.now()/1000 - task.started_at;
                                const avgTime = elapsed / completed;
                                const remaining = Math.round(avgTime * (task.total_pdfs - completed));
                                const hours = Math.floor(remaining / 3600);
                                const minutes = Math.floor((remaining % 3600) / 60);
                                
                                if (hours > 0 || minutes > 0) {
                                    html += `<br>⏱️ <strong>আনুমানিক বাকি:</strong> `;
                                    if (hours > 0) html += `${hours} ঘন্টা `;
                                    html += `${minutes} মিনিট`;
                                }
                            }
                            html += `</div>`;
                        }
                        
                        if (task.completed_at) {
                            html += `<br>✅ সমাপ্ত: ${new Date(task.completed_at * 1000).toLocaleString('bn-BD')}`;
                        }
                        
                        html += `</div></div>`;
                    }
                    tasksDiv.innerHTML = html;
                }
            } catch (e) {
                document.getElementById('tasksList').innerHTML = '<p style="color: red;">টাস্ক লোড করতে ব্যর্থ</p>';
            }
        }

        function fillTafsir() {
            document.getElementById('inputData').value = `তাফসীর ফী যিলালিল কোরআন
https://archive.org/download/20260415_20260415_0945/1.pdf`;
        }

        function fillTafsirFull() {
            let urls = 'তাফসীর ফী যিলালিল কোরআন (২২ খন্ড)\\n';
            for (let i = 1; i <= 22; i++) {
                urls += `https://archive.org/download/20260415_20260415_0945/${i}.pdf\\n`;
            }
            document.getElementById('inputData').value = urls;
        }

        function fillTafsirItem() {
            document.getElementById('inputData').value = `তাফসীর ফী যিলালিল কোরআন(২২ খন্ড)
https://archive.org/details/20260415_20260415_0945`;
        }

        function clearForm() {
            document.getElementById('inputData').value = '';
            document.getElementById('result').className = 'result';
            document.getElementById('result').innerHTML = '';
        }

        async function previewOnly() {
            const inputData = document.getElementById('inputData').value;
            const resultDiv = document.getElementById('result');
            
            const lines = inputData.split('\\n').filter(l => l.trim());
            if (lines.length < 2) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '❌ অন্তত একটি বইয়ের নাম এবং একটি লিংক দিন';
                return;
            }

            const bookName = lines[0];
            const urls = lines.slice(1).filter(l => l.startsWith('http'));
            
            let html = `<strong>📚 বই:</strong> ${bookName}<br>`;
            html += `<strong>📄 লিংক সংখ্যা:</strong> ${urls.length}<br>`;
            
            resultDiv.className = 'result success';
            resultDiv.innerHTML = html;
        }

        document.getElementById('processForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const inputData = document.getElementById('inputData').value;
            const resultDiv = document.getElementById('result');
            
            const lines = inputData.split('\\n').filter(l => l.trim());
            if (lines.length < 2) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '❌ অন্তত একটি বইয়ের নাম এবং একটি লিংক দিন';
                return;
            }

            const bookName = lines[0];
            const urls = lines.slice(1).filter(l => l.startsWith('http'));
            
            if (urls.length === 0) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '❌ কোনো বৈধ লিংক পাওয়া যায়নি';
                return;
            }

            resultDiv.className = 'result success';
            resultDiv.innerHTML = '⏳ প্রসেসিং শুরু হচ্ছে...';

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ book_name: bookName, urls: urls })
                });
                
                const data = await response.json();
                
                if (data.status === 'started') {
                    resultDiv.innerHTML = `✅ প্রসেসিং শুরু হয়েছে!<br>
                        📁 বই: ${bookName}<br>
                        📄 লিংক সংখ্যা: ${data.pdf_count}<br>
                        🆔 টাস্ক ID: ${data.task_id}`;
                    loadTasks();
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `❌ ত্রুটি: ${data.message}`;
                }
            } catch (err) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `❌ সংযোগ ত্রুটি`;
            }
        });

        loadTasks();
        setInterval(loadTasks, 3000);
    </script>
</body>
</html>
"""

# ============ FastAPI অ্যাপ ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global shutdown_in_progress
    shutdown_in_progress = True

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.head("/")
async def root_head():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/check-folder/{folder_name}")
async def check_folder(folder_name: str):
    clean_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    progress = get_hf_folder_progress(clean_name)
    return progress

@app.get("/memory")
async def get_memory():
    """মেমরি ব্যবহার API"""
    return {
        "process_mb": get_memory_usage(),
        "system": get_system_memory()
    }

@app.post("/process")
async def process_form(request: Request):
    try:
        data = await request.json()
        book_name = data.get('book_name', f"tafsir_{int(time.time())}")
        urls = data.get('urls', [])

        if not urls:
            return {"status": "error", "message": "কোনো লিংক দেওয়া হয়নি"}

        pdf_urls = []
        for url in urls:
            if 'archive.org/details/' in url:
                item_id = url.split('/')[-1]
                for i in range(1, 23):
                    pdf_urls.append(f"https://archive.org/download/{item_id}/{i}.pdf")
            else:
                pdf_urls.append(url)

        if not pdf_urls:
            return {"status": "error", "message": "কোনো বৈধ PDF লিংক পাওয়া যায়নি"}

        def extract_number(url):
            match = re.search(r'/(\d+)\.pdf', url)
            return int(match.group(1)) if match else 999
        pdf_urls.sort(key=extract_number)

        task_id = f"web_{int(time.time())}"
        with running_tasks_lock:
            running_tasks[task_id] = {
                "status": "running",
                "started_at": time.time(),
                "folder_name": book_name,
                "total_pdfs": len(pdf_urls),
                "completed_pdfs": 0,
                "current_pdf": None,
                "current_pdf_index": 0,
                "current_pdf_total_pages": 0,
                "current_page": 0,
                "memory_usage": get_memory_usage(),
                "system_memory": get_system_memory()
            }
        with task_controls_lock:
            task_controls[task_id] = {"cancel": False}

        thread = threading.Thread(
            target=process_pdf_urls_sync,
            args=(pdf_urls, book_name, task_id),
            daemon=True
        )
        thread.start()
        print(f"[DEBUG] Thread started for task {task_id}", flush=True)

        return {
            "status": "started",
            "book_name": book_name,
            "task_id": task_id,
            "pdf_count": len(pdf_urls)
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/tasks")
async def get_tasks():
    with running_tasks_lock:
        tasks_copy = {}
        for task_id, task_info in running_tasks.items():
            task_copy = task_info.copy()

            if task_info.get("folder_name"):
                clean_name = task_info["folder_name"].replace(' ', '_')
                checkpoint = load_checkpoint_sync(f"direct_{clean_name}")

                processed = checkpoint.get('processed', [])
                current = checkpoint.get('current')
                last_page = checkpoint.get('last_page', 0)

                task_copy["completed_pdfs"] = len(processed)

                if task_copy["total_pdfs"] > 0:
                    task_copy["percent"] = round(len(processed) * 100 / task_copy["total_pdfs"], 1)

                if current and last_page > 0:
                    task_copy["current_pdf"] = current
                    task_copy["current_page"] = last_page

            # 🔥 মেমরি ইনফো সবসময় আপডেট
            task_copy["memory_usage"] = get_memory_usage()
            task_copy["system_memory"] = get_system_memory()

            tasks_copy[task_id] = task_copy

        return tasks_copy

@app.get("/cancel/{task_id}")
async def cancel_task(task_id: str):
    with task_controls_lock:
        if task_id in task_controls:
            task_controls[task_id]["cancel"] = True
            return {"status": "cancelled"}
    return {"status": "not found"}

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)