#!/usr/bin/env python3
"""
FastAPI + HTML Form (No File System)
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
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import fitz  # PyMuPDF
from huggingface_hub import HfApi, upload_file

# ============ কনফিগারেশন ============
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")

TEMP_DIR = Path("/tmp/tafsir_temp")
TEMP_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = Path("/tmp/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MIN_FREE_SPACE_MB = 500
UPLOAD_CONCURRENCY = 3
MAX_JOB_RUNTIME = 6 * 3600
PAGE_HASH_CACHE_SIZE = 10000
# =====================================

# গ্লোবাল ভেরিয়েবল
running_tasks = {}
running_tasks_lock = asyncio.Lock()
task_controls = {}
task_controls_lock = asyncio.Lock()
upload_semaphore = asyncio.Semaphore(UPLOAD_CONCURRENCY)
page_hash_cache = OrderedDict()
page_hash_cache_lock = asyncio.Lock()
shutdown_in_progress = False

class ProcessRequest(BaseModel):
    book_name: str
    urls: list

# ============ Threading Helper ============
def run_async_in_thread(coro, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro(*args))
    except Exception as e:
        print(f"[Thread] Error: {e}", flush=True)
        traceback.print_exc()
    finally:
        loop.close()

# ============ Disk Space Check ============
async def check_disk_space():
    try:
        total, used, free = shutil.disk_usage("/tmp")
        free_mb = free // (1024 * 1024)
        if free_mb < MIN_FREE_SPACE_MB:
            raise RuntimeError(f"Low disk space: {free_mb} MB free")
        return True
    except Exception as e:
        print(f"[Disk Check] Warning: {e}", flush=True)
        return True

# ============ LRU Cache for Page Hash ============
async def is_page_already_uploaded(page_hash):
    async with page_hash_cache_lock:
        if page_hash in page_hash_cache:
            page_hash_cache.move_to_end(page_hash)
            return True
        return False

async def mark_page_uploaded_cache(page_hash):
    async with page_hash_cache_lock:
        if page_hash not in page_hash_cache:
            page_hash_cache[page_hash] = time.time()
            if len(page_hash_cache) > PAGE_HASH_CACHE_SIZE:
                page_hash_cache.popitem(last=False)

# ============ Checkpoint ============
def get_checkpoint_path(folder_key):
    safe_key = folder_key.replace('/', '_').replace('\\', '_')
    return CHECKPOINT_DIR / f"checkpoint_{safe_key}.json"

async def async_save_checkpoint(folder_key, checkpoint):
    try:
        checkpoint_path = get_checkpoint_path(folder_key)
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        os.replace(temp_path, checkpoint_path)
    except Exception as e:
        print(f"[Checkpoint] Error: {e}", flush=True)

async def load_checkpoint(folder_key):
    try:
        checkpoint_path = get_checkpoint_path(folder_key)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"processed": [], "current": None, "last_page": 0, "last_page_map": {}}

# ============ Internet Archive ============
async def extract_pdfs_from_archive_item(item_url: str, max_pdfs: int = 30):
    pdf_urls = []
    try:
        item_match = re.search(r'/details/([^/?]+)', item_url)
        if not item_match:
            return pdf_urls
        item_id = item_match.group(1)
        print(f"[Archive] Extracting from: {item_id}", flush=True)
        
        # প্রথমে নাম্বার অনুযায়ী ট্রাই করুন (সবচেয়ে কমন প্যাটার্ন)
        async with httpx.AsyncClient(timeout=10.0) as client:
            for i in range(1, max_pdfs + 1):
                test_url = f"https://archive.org/download/{item_id}/{i}.pdf"
                try:
                    response = await client.head(test_url)
                    if response.status_code == 200:
                        pdf_urls.append(test_url)
                except:
                    pass
        
        # যদি না পাওয়া যায়, মেটাডেটা API ট্রাই করুন
        if not pdf_urls:
            metadata_url = f"https://archive.org/metadata/{item_id}"
            response = await client.get(metadata_url)
            if response.status_code == 200:
                data = response.json()
                for file_info in data.get('files', []):
                    file_name = file_info.get('name', '')
                    if file_name.lower().endswith('.pdf'):
                        pdf_urls.append(f"https://archive.org/download/{item_id}/{file_name}")
    except Exception as e:
        print(f"[Archive] Error: {e}", flush=True)
    
    return pdf_urls

# ============ PDF ডাউনলোড ============
async def download_pdf_stream(url):
    print(f"[DEBUG] Downloading: {url[:80]}...", flush=True)
    await check_disk_space()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Referer': 'https://archive.org/'
        }
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True, headers=headers) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
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

async def upload_to_hf_with_retry(folder_path, file_path, img_name, page_hash):
    if await is_page_already_uploaded(page_hash):
        return True
    for retry in range(3):
        try:
            path_in_repo = f"{folder_path}/{img_name}"
            with open(file_path, 'rb') as f:
                async with upload_semaphore:
                    upload_file(
                        path_or_fileobj=f,
                        path_in_repo=path_in_repo,
                        repo_id=HF_DATASET,
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
            await mark_page_uploaded_cache(page_hash)
            try:
                os.unlink(file_path)
            except:
                pass
            return True
        except Exception as e:
            if retry == 2:
                raise
            wait_time = 2 ** retry
            print(f"Upload retry {retry+1}/3: {e}", flush=True)
            await asyncio.sleep(wait_time)
    return False

def is_page_uploaded_checkpoint(checkpoint, pdf_name, page_num):
    last_page_map = checkpoint.get('last_page_map', {})
    last_page = last_page_map.get(pdf_name, 0)
    return page_num <= last_page

def mark_page_uploaded_checkpoint(checkpoint, pdf_name, page_num):
    if 'last_page_map' not in checkpoint:
        checkpoint['last_page_map'] = {}
    current = checkpoint['last_page_map'].get(pdf_name, 0)
    if page_num > current:
        checkpoint['last_page_map'][pdf_name] = page_num

async def is_task_cancelled(task_id):
    async with task_controls_lock:
        return task_controls.get(task_id, {}).get("cancel", False)

async def process_single_pdf(pdf, clean_folder_name, folder_key, checkpoint, task_id):
    sub_folder = str(pdf['number'])
    full_hf_path = f"{clean_folder_name}/{sub_folder}"
    print(f"[INFO] Processing: {pdf['name']} -> {full_hf_path}", flush=True)

    start_page = 0
    if checkpoint.get('current') == pdf['name']:
        start_page = checkpoint.get('last_page', 0)

    if not pdf.get('download_link'):
        return 0, 0, False

    print(f"[INFO] Downloading: {pdf['name']}", flush=True)
    try:
        pdf_path = await download_pdf_stream(pdf['download_link'])
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", flush=True)
        return 0, 0, False

    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"[INFO] {pdf['name']}: {total_pages} pages", flush=True)

        pages_processed = 0
        for page_num in range(start_page, total_pages):
            if shutdown_in_progress or await is_task_cancelled(task_id):
                return pages_processed, total_pages, True

            if is_page_uploaded_checkpoint(checkpoint, pdf['name'], page_num + 1):
                continue

            page = doc.load_page(page_num)
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_bytes = pix.tobytes("png")
            page_hash = hashlib.md5(img_bytes).hexdigest()

            if await is_page_already_uploaded(page_hash):
                mark_page_uploaded_checkpoint(checkpoint, pdf['name'], page_num + 1)
                continue

            img_name = f"page_{page_num+1:04d}.png"
            tmp_path = TEMP_DIR / f"{task_id}_{page_num}.png"

            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)

            try:
                await upload_to_hf_with_retry(full_hf_path, tmp_path, img_name, page_hash)
                mark_page_uploaded_checkpoint(checkpoint, pdf['name'], page_num + 1)
                pages_processed += 1
            except Exception as e:
                print(f"[ERROR] Upload failed page {page_num+1}: {e}", flush=True)

            if page_num % 50 == 0:
                checkpoint['current'] = pdf['name']
                checkpoint['last_page'] = page_num
                await async_save_checkpoint(folder_key, checkpoint)
                print(f"[INFO] Progress: {page_num+1}/{total_pages} pages", flush=True)

        checkpoint['current'] = None
        checkpoint['last_page'] = total_pages
        await async_save_checkpoint(folder_key, checkpoint)

        return pages_processed, total_pages, False
    finally:
        if doc:
            doc.close()
        try:
            os.unlink(pdf_path)
        except:
            pass

async def process_pdf_urls(pdf_urls: list, folder_name: str, task_id: str):
    print(f"[DEBUG] ========== PROCESS STARTED ==========", flush=True)
    print(f"[DEBUG] Folder: {folder_name}", flush=True)
    print(f"[DEBUG] PDF Count: {len(pdf_urls)}", flush=True)

    try:
        clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
        print(f"[INFO] Starting: {clean_folder_name}", flush=True)

        checkpoint = await load_checkpoint(f"direct_{clean_folder_name}")
        already_processed = set(checkpoint.get('processed', []))
        processed = []

        for idx, url in enumerate(pdf_urls):
            if shutdown_in_progress or await is_task_cancelled(task_id):
                break

            name_match = re.search(r'/(\d+)\.pdf', url)
            pdf_number = int(name_match.group(1)) if name_match else (idx + 1)
            pdf_name = f"{pdf_number}.pdf"

            if pdf_name in already_processed:
                print(f"[INFO] Skipping {pdf_name}", flush=True)
                continue

            pdf = {'name': pdf_name, 'number': pdf_number, 'download_link': url}
            print(f"[INFO] [{idx+1}/{len(pdf_urls)}] Starting {pdf_name}", flush=True)

            try:
                pages_processed, total_pages, cancelled = await process_single_pdf(
                    pdf, clean_folder_name, f"direct_{clean_folder_name}", checkpoint, task_id
                )
                if cancelled:
                    break

                processed.append(pdf_name)
                checkpoint['processed'] = list(set(checkpoint.get('processed', []) + [pdf_name]))
                await async_save_checkpoint(f"direct_{clean_folder_name}", checkpoint)
                print(f"[INFO] Completed: {pdf_name} ({total_pages} pages)", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed: {pdf_name} - {e}", flush=True)
                traceback.print_exc()

        if processed:
            print(f"[INFO] All completed! {len(processed)} PDFs processed", flush=True)

        async with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "completed"
                running_tasks[task_id]["completed_at"] = time.time()
    except Exception as e:
        print(f"[DEBUG] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        async with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error"] = str(e)

# ============ HTML টেমপ্লেট (সরাসরি স্ট্রিং হিসেবে) ============
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
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #1a5f7a;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        input, textarea {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #1a5f7a;
        }
        textarea {
            min-height: 200px;
            font-family: monospace;
        }
        .btn {
            background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.02);
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .btn-secondary {
            background: #6c757d;
            flex: 1;
        }
        .btn-primary {
            background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%);
            flex: 2;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            display: none;
        }
        .result.success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            display: block;
        }
        .result.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            display: block;
        }
        .info-box {
            background: #e7f3ff;
            border: 1px solid #b8daff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .info-box h3 {
            color: #004085;
            margin-bottom: 10px;
        }
        .info-box ul {
            margin-left: 20px;
            color: #004085;
        }
        .quick-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .quick-btn {
            background: #e9ecef;
            border: 1px solid #ced4da;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
        }
        .quick-btn:hover {
            background: #dee2e6;
        }
        .status-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-running { background: #fff3cd; color: #856404; }
        .status-completed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 তাফসীর PDF প্রসেসর</h1>
        <p class="subtitle">PDF থেকে ইমেজ কনভার্ট করে Hugging Face Dataset-এ আপলোড করুন</p>
        
        <div class="info-box">
            <h3>📋 ব্যবহার নির্দেশিকা</h3>
            <ul>
                <li>প্রথম লাইনে বইয়ের নাম লিখুন (ইংরেজি বা বাংলা)</li>
                <li>তারপর প্রতি লাইনে একটি করে PDF লিংক দিন</li>
                <li>Internet Archive আইটেম লিংক দিলে অটোমেটিক সব PDF বের হবে</li>
            </ul>
        </div>

        <div class="quick-buttons">
            <span class="quick-btn" onclick="fillExample()">📝 উদাহরণ</span>
            <span class="quick-btn" onclick="fillTafsir()">📖 তাফসীর (1 PDF)</span>
            <span class="quick-btn" onclick="fillTafsirFull()">📚 তাফসীর (22 PDF)</span>
            <span class="quick-btn" onclick="clearForm()">🗑️ ক্লিয়ার</span>
        </div>

        <form id="processForm">
            <div class="form-group">
                <label for="inputData">📄 বইয়ের নাম ও PDF লিংকসমূহ:</label>
                <textarea id="inputData" name="inputData" placeholder="উদাহরণ:
তাফসীর ফী যিলালিল কোরআন
https://archive.org/download/20260415_20260415_0945/1.pdf
https://archive.org/download/20260415_20260415_0945/2.pdf"></textarea>
            </div>
            
            <div class="btn-group">
                <button type="button" class="btn btn-secondary" onclick="previewOnly()">👁️ প্রিভিউ</button>
                <button type="submit" class="btn btn-primary">🚀 প্রসেসিং শুরু করুন</button>
            </div>
        </form>

        <div id="result" class="result"></div>
        
        <div style="margin-top: 20px;">
            <h3>📊 চলমান টাস্ক</h3>
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
                    tasksDiv.innerHTML = '<p>কোনো চলমান টাস্ক নেই</p>';
                } else {
                    let html = '';
                    for (const [id, task] of Object.entries(tasks)) {
                        const statusClass = task.status === 'running' ? 'status-running' : 
                                          (task.status === 'completed' ? 'status-completed' : 'status-failed');
                        html += `<div style="background: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 8px;">`;
                        html += `<strong>${id}</strong> <span class="status-badge ${statusClass}">${task.status}</span><br>`;
                        html += `শুরু: ${new Date(task.started_at * 1000).toLocaleString('bn-BD')}`;
                        if (task.completed_at) {
                            html += `<br>সমাপ্ত: ${new Date(task.completed_at * 1000).toLocaleString('bn-BD')}`;
                        }
                        html += `</div>`;
                    }
                    tasksDiv.innerHTML = html;
                }
            } catch (e) {
                document.getElementById('tasksList').innerHTML = '<p>টাস্ক লোড করতে ব্যর্থ</p>';
            }
        }

        function fillExample() {
            document.getElementById('inputData').value = `তাফসীর টেস্ট
https://archive.org/download/20260415_20260415_0945/1.pdf`;
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
                resultDiv.innerHTML = '❌ অন্তত একটি বইয়ের নাম এবং একটি PDF লিংক দিন';
                return;
            }

            const bookName = lines[0];
            const urls = lines.slice(1).filter(l => l.startsWith('http'));
            
            let html = `<strong>📚 বই:</strong> ${bookName}<br>`;
            html += `<strong>📄 PDF সংখ্যা:</strong> ${urls.length}<br><br>`;
            html += `<strong>লিংকসমূহ:</strong><br>`;
            urls.slice(0, 5).forEach(url => html += `• ${url}<br>`);
            if (urls.length > 5) html += `... এবং আরো ${urls.length - 5}টি<br>`;
            
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
                resultDiv.innerHTML = '❌ অন্তত একটি বইয়ের নাম এবং একটি PDF লিংক দিন';
                return;
            }

            const bookName = lines[0];
            const urls = lines.slice(1).filter(l => l.startsWith('http'));
            
            if (urls.length === 0) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '❌ কোনো বৈধ PDF লিংক পাওয়া যায়নি';
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
                    resultDiv.innerHTML = `✅ প্রসেসিং শুরু হয়েছে!<br>
                        📁 বই: ${bookName}<br>
                        📄 PDF সংখ্যা: ${urls.length}<br>
                        🆔 টাস্ক ID: ${data.task_id}<br><br>
                        <a href="#" onclick="loadTasks(); return false;">টাস্ক স্ট্যাটাস রিফ্রেশ করুন</a>`;
                    loadTasks();
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `❌ ত্রুটি: ${data.message}`;
                }
            } catch (err) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `❌ সংযোগ ত্রুটি: ${err.message}`;
            }
        });

        loadTasks();
        setInterval(loadTasks, 5000);
    </script>
</body>
</html>
"""

# ============ FastAPI অ্যাপ ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    shutdown_in_progress = True

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def home():
    """HTML ফর্ম দেখান"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/process")
async def process_form(request: Request):
    """ফর্ম থেকে ডাটা নিয়ে প্রসেসিং শুরু করুন"""
    try:
        data = await request.json()
        book_name = data.get('book_name', f"tafsir_{int(time.time())}")
        urls = data.get('urls', [])
        
        if not urls:
            return {"status": "error", "message": "কোনো PDF লিংক দেওয়া হয়নি"}
        
        # Internet Archive আইটেম চেক
        pdf_urls = []
        for url in urls:
            if 'archive.org/details/' in url:
                extracted = await extract_pdfs_from_archive_item(url)
                pdf_urls.extend(extracted)
            else:
                pdf_urls.append(url)
        
        if not pdf_urls:
            return {"status": "error", "message": "কোনো বৈধ PDF লিংক পাওয়া যায়নি"}
        
        # নাম্বার অনুযায়ী সর্ট
        def extract_number(url):
            match = re.search(r'/(\d+)\.pdf', url)
            return int(match.group(1)) if match else 999
        pdf_urls.sort(key=extract_number)
        
        task_id = f"web_{int(time.time())}"
        async with running_tasks_lock:
            running_tasks[task_id] = {"status": "running", "started_at": time.time()}
        async with task_controls_lock:
            task_controls[task_id] = {"cancel": False}
        
        thread = threading.Thread(
            target=run_async_in_thread,
            args=(process_pdf_urls, pdf_urls, book_name, task_id),
            daemon=True
        )
        thread.start()
        
        return {
            "status": "started",
            "message": f"Processing {len(pdf_urls)} PDFs",
            "book_name": book_name,
            "task_id": task_id,
            "pdf_count": len(pdf_urls)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-process")
async def test_process():
    """প্রসেসিং টেস্ট করুন"""
    try:
        url = "https://archive.org/download/20260415_20260415_0945/1.pdf"
        print(f"[TEST] Downloading: {url}", flush=True)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("GET", url) as response:
                async for chunk in response.aiter_bytes():
                    temp_file.write(chunk)
        temp_file.close()
        
        size = os.path.getsize(temp_file.name)
        os.unlink(temp_file.name)
        
        return {"status": "success", "downloaded_bytes": size}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/tasks")
async def get_tasks():
    """চলমান টাস্ক দেখুন"""
    async with running_tasks_lock:
        return {k: v for k, v in running_tasks.items()}

@app.get("/test-connection")
async def test_connection():
    """Internet Archive সংযোগ টেস্ট"""
    try:
        import socket
        import requests
        
        # DNS টেস্ট
        ip = socket.gethostbyname("archive.org")
        
        # HTTP টেস্ট
        response = requests.get("https://archive.org", timeout=10)
        
        return {
            "status": "success",
            "archive.org_ip": ip,
            "http_status": response.status_code
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
        

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    async with task_controls_lock:
        if task_id in task_controls:
            task_controls[task_id]["cancel"] = True
            return {"status": "cancelled"}
    return {"status": "not found"}

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
