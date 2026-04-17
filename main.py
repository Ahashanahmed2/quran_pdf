#!/usr/bin/env python3
"""
একক ফাইল: FastAPI + Telegram Bot (Production Ready - ALL Issues Fixed with Proper Thread-Local Storage)
"""

import os
import io
import re
import json
import time
import hashlib
import asyncio
import threading
import tempfile
import traceback
import queue as queue_module
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from queue import Queue, Full as QueueFull
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import httpx
import fitz  # PyMuPDF
from mediafire import MediaFireApi
from huggingface_hub import HfApi, upload_file

# ============ কনফিগারেশন ============
MEDIAFIRE_EMAIL = os.environ.get("MEDIAFIRE_EMAIL")
MEDIAFIRE_PASSWORD = os.environ.get("MEDIAFIRE_PASSWORD")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

TEMP_DIR = Path("/tmp/tafsir_temp")
TEMP_DIR.mkdir(exist_ok=True)

# Per-job checkpoint file
CHECKPOINT_DIR = Path("/tmp/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Startup cleanup
for f in TEMP_DIR.glob("*"):
    try:
        if f.is_file():
            f.unlink()
    except:
        pass

for f in CHECKPOINT_DIR.glob("checkpoint_*.json"):
    try:
        # Only clean old checkpoints (> 7 days)
        if time.time() - f.stat().st_mtime > 7 * 86400:
            f.unlink()
    except:
        pass

# গ্লোবাল ভেরিয়েবল
application = None
bot = None
mediafire_api = None
mediafire_session = None
http_client = None
polling_task = None
main_loop = None

# Task management (with per-task executors)
running_tasks = {}
running_tasks_lock = threading.Lock()
task_controls = {}
task_controls_lock = threading.Lock()
task_executors = {}
task_executors_lock = threading.Lock()

# Rate limit control
upload_semaphore = threading.Semaphore(3)
# Checkpoint lock (thread-safe)
checkpoint_lock = threading.Lock()
# Session lock (for global fallback)
session_lock = threading.Lock()
# Telegram throttling
last_telegram_sent = {}
telegram_lock = threading.Lock()
telegram_queue = Queue(maxsize=1000)
telegram_worker_running = False
telegram_worker_thread = None

# Thread-local storage for MediaFire sessions (CORRECT way)
thread_local = threading.local()

# Checkpoint batch save
checkpoint_batch_counter = {}
checkpoint_batch_lock = threading.Lock()

# Disk write queue for checkpoints
checkpoint_write_queue = queue_module.Queue()
checkpoint_writer_running = False

# Dynamic thread pool size per task
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = min(8, CPU_COUNT * 2)

# Job timeout (6 hours)
MAX_JOB_RUNTIME = 6 * 3600
# =====================================

class ProcessRequest(BaseModel):
    folder_key: str
    folder_name: str
    telegram_chat_id: int

# ============ Checkpoint Async Writer ============
def checkpoint_writer_worker():
    """Background worker for async checkpoint writes"""
    while checkpoint_writer_running:
        try:
            item = checkpoint_write_queue.get(timeout=1)
            if item is None:
                break
            folder_key, checkpoint = item
            with checkpoint_lock:
                checkpoint_path = get_checkpoint_path(folder_key)
                temp_path = checkpoint_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                temp_path.rename(checkpoint_path)
            checkpoint_write_queue.task_done()
        except Exception as e:
            print(f"[Checkpoint Writer] Error: {e}")

def async_save_checkpoint(folder_key, checkpoint):
    """Non-blocking checkpoint save"""
    try:
        checkpoint_write_queue.put((folder_key, checkpoint.copy()), timeout=1)
    except queue_module.Full:
        print(f"[Checkpoint] Queue full, sync saving {folder_key}")
        with checkpoint_lock:
            checkpoint_path = get_checkpoint_path(folder_key)
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            temp_path.rename(checkpoint_path)

# ============ Telegram Queue Worker ============
def telegram_worker():
    """Telegram message queue worker (with retry and error logging)"""
    while telegram_worker_running:
        try:
            item = telegram_queue.get(timeout=1)
            if item is None:
                break
            chat_id, message = item
            # Retry logic at queue worker level
            for retry in range(3):
                try:
                    send_telegram_direct(chat_id, message)
                    break
                except Exception as e:
                    print(f"[Telegram Worker] Retry {retry+1}/3 error: {e}")
                    time.sleep(2)
            telegram_queue.task_done()
            time.sleep(0.5)
        except Exception as e:
            print(f"[Telegram Worker] Unexpected error: {e}")

def send_telegram_direct(chat_id, message):
    """Direct telegram send (without queue)"""
    if not TELEGRAM_BOT_TOKEN:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": chat_id, "text": message[:4000], "parse_mode": "Markdown"}
        response = httpx.post(url, json=data, timeout=10)
        if response.status_code != 200:
            print(f"[Telegram] HTTP {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"[Telegram] Send error: {e}")
        raise

def send_telegram(chat_id, message):
    """Queue-based telegram send (with bounded queue and fallback)"""
    try:
        telegram_queue.put((chat_id, message), timeout=1)
    except QueueFull:
        # Fallback: try direct send
        print(f"[Telegram] Queue full, trying direct send for chat {chat_id}")
        try:
            send_telegram_direct(chat_id, message)
        except Exception as e:
            print(f"[Telegram] Direct send also failed: {e}")
    except Exception as e:
        print(f"[Telegram] Queue error: {e}")

def throttled_send(chat_id, message, interval=2):
    """Rate limit সহ টেলিগ্রাম মেসেজ"""
    with telegram_lock:
        now = time.time()
        last_time = last_telegram_sent.get(chat_id, 0)
        if now - last_time >= interval:
            send_telegram(chat_id, message)
            last_telegram_sent[chat_id] = now
            return True
        return False

# ============ Retry Decorator ============
def retry(max_retries=3, delay=2, refresh_session=False):
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if refresh_session and i < max_retries - 1:
                        refresh_mediafire_session()
                    if i == max_retries - 1:
                        raise
                    print(f"Retry {i+1}/{max_retries} for {func.__name__}: {e}")
                    await asyncio.sleep(delay)
            return None
        
        def sync_wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if refresh_session and i < max_retries - 1 and ("expired" in error_msg or "invalid" in error_msg):
                        refresh_mediafire_session()
                    if i == max_retries - 1:
                        raise
                    print(f"Retry {i+1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(delay)
            return None
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

# ============ MediaFire Session Management ============
def refresh_mediafire_session():
    """MediaFire সেশন রিফ্রেশ করে (global fallback)"""
    global mediafire_api, mediafire_session
    with session_lock:
        try:
            mediafire_api = MediaFireApi()
            mediafire_session = mediafire_api.user_get_session_token(
                email=MEDIAFIRE_EMAIL, 
                password=MEDIAFIRE_PASSWORD, 
                app_id='42511'
            )
            mediafire_api.session = mediafire_session
            print("🔄 MediaFire global session refreshed")
        except Exception as e:
            print(f"⚠️ Failed to refresh MediaFire session: {e}")

def get_mediafire_session():
    """MediaFire সেশন (thread-local - CORRECT implementation)"""
    # Get thread-local session
    if hasattr(thread_local, 'mediafire_api') and hasattr(thread_local, 'mediafire_session'):
        return thread_local.mediafire_api
    
    # Create new session for this thread
    try:
        api = MediaFireApi()
        session = api.user_get_session_token(
            email=MEDIAFIRE_EMAIL, 
            password=MEDIAFIRE_PASSWORD, 
            app_id='42511'
        )
        api.session = session
        thread_local.mediafire_api = api
        thread_local.mediafire_session = session
        return api
    except Exception as e:
        print(f"⚠️ Failed to create thread-local MediaFire session: {e}")
        # Fallback to global
        global mediafire_api
        if mediafire_api is None:
            refresh_mediafire_session()
        return mediafire_api

# ============ টেলিগ্রাম ফাংশন ============
def extract_from_mediafire_url(url):
    """MediaFire URL থেকে ফোল্ডার কী ও নাম বের করে"""
    key_match = re.search(r'/folder/([a-zA-Z0-9]+)', url)
    if not key_match:
        return None, None
    folder_key = key_match.group(1)
    name_match = re.search(r'/folder/[a-zA-Z0-9]+/(.+?)$', url)
    if name_match:
        folder_name = name_match.group(1).replace('+', ' ').replace('%20', ' ')
    else:
        folder_name = folder_key
    return folder_key, folder_name

# ============ PDF প্রসেসিং ফাংশন ============
def get_checkpoint_path(folder_key):
    """Per-job checkpoint file path"""
    safe_key = folder_key.replace('/', '_').replace('\\', '_')
    return CHECKPOINT_DIR / f"checkpoint_{safe_key}.json"

def load_checkpoint(folder_key):
    with checkpoint_lock:
        checkpoint_path = get_checkpoint_path(folder_key)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
    return {"processed": [], "current": None, "last_page": 0, "last_page_map": {}}

def is_page_uploaded(checkpoint, pdf_name, page_num):
    """পৃষ্ঠা আগে আপলোড হয়েছে কিনা চেক করে"""
    last_page_map = checkpoint.get('last_page_map', {})
    last_page = last_page_map.get(pdf_name, 0)
    return page_num <= last_page

def mark_page_uploaded(checkpoint, pdf_name, page_num):
    """পৃষ্ঠা আপলোড হয়েছে মার্ক করে"""
    if 'last_page_map' not in checkpoint:
        checkpoint['last_page_map'] = {}
    current = checkpoint['last_page_map'].get(pdf_name, 0)
    if page_num > current:
        checkpoint['last_page_map'][pdf_name] = page_num

def is_task_cancelled(task_id):
    """টাস্ক ক্যান্সেল হয়েছে কিনা চেক করে"""
    with task_controls_lock:
        return task_controls.get(task_id, {}).get("cancel", False)

def get_task_executor(task_id):
    """Get or create per-task executor"""
    with task_executors_lock:
        if task_id not in task_executors:
            task_executors[task_id] = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        return task_executors[task_id]

def cleanup_task_executor(task_id):
    """Cleanup per-task executor"""
    with task_executors_lock:
        if task_id in task_executors:
            executor = task_executors[task_id]
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except:
                pass
            del task_executors[task_id]

def download_pdf_stream(url):
    """Stream PDF download to temp file (memory efficient)"""
    import requests
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except:
            pass
        raise e

@retry(max_retries=3, delay=2)
def upload_to_hf_with_retry(folder_path, file_path, img_name):
    """Temp file থেকে HF আপলোড (with retry)"""
    try:
        path_in_repo = f"{folder_path}/{img_name}"
        with open(file_path, 'rb') as f:
            upload_file(
                path_or_fileobj=f,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET,
                repo_type="dataset",
                token=HF_TOKEN
            )
        return True
    finally:
        try:
            os.unlink(file_path)
        except:
            pass

@retry(max_retries=3, delay=2, refresh_session=True)
def get_mediafire_files(folder_key):
    """MediaFire থেকে সব PDF-এর লিংক বের করে"""
    api = get_mediafire_session()
    folder_content = api.folder_get_content(folder_key=folder_key)
    files = []
    for item in folder_content['folder_content']:
        if item['type'] == 'file' and item['filename'].endswith('.pdf'):
            file_links = api.file_get_links(quickkey=item['quickkey'])
            download_link = None
            for link in file_links['links']:
                if link['type'] == 'normal_download':
                    download_link = link['normal_download']
                    break
            try:
                num = int(item['filename'].replace('.pdf', ''))
            except:
                num = 999
            files.append({
                'name': item['filename'],
                'number': num,
                'download_link': download_link,
            })
    files.sort(key=lambda x: x['number'])
    return files

def process_single_pdf(pdf, clean_folder_name, folder_key, chat_id, checkpoint, task_id):
    """একটি PDF প্রসেস করে (with cancel support)"""
    sub_folder = str(pdf['number'])
    full_hf_path = f"{clean_folder_name}/{sub_folder}"
    
    throttled_send(chat_id, f"📄 *প্রসেসিং: {pdf['name']}*\n📁 লোকেশন: `{full_hf_path}`")
    
    # Resume logic
    start_page = 0
    if checkpoint.get('current') == pdf['name']:
        start_page = checkpoint.get('last_page', 0)
        if start_page > 0:
            throttled_send(chat_id, f"📌 *Resuming from page {start_page + 1}*")
    
    # Stream PDF download to temp file
    pdf_path = download_pdf_stream(pdf['download_link'])
    
    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        throttled_send(chat_id, f"🖼️ {total_pages} পৃষ্ঠা কনভার্ট ও আপলোড শুরু...")
        
        BATCH_SIZE = 5
        batch_futures = []
        pages_processed = 0
        checkpoint_counter = 0
        job_start_time = time.time()
        last_checkpoint_save = time.time()
        
        for page_num in range(start_page, total_pages):
            if is_task_cancelled(task_id):
                throttled_send(chat_id, f"⛔ *Task cancelled by user* at page {page_num + 1}")
                return pages_processed, total_pages, True
            
            if time.time() - job_start_time > MAX_JOB_RUNTIME:
                throttled_send(chat_id, f"⏰ *Job timeout* after {MAX_JOB_RUNTIME/3600} hours")
                return pages_processed, total_pages, False
            
            if is_page_uploaded(checkpoint, pdf['name'], page_num + 1):
                continue
                
            page = doc.load_page(page_num)
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img_name = f"page_{page_num+1:04d}.png"
            tmp_path = TEMP_DIR / f"{task_id}_{page_num}_{int(time.time()*1000)}.png"
            pix.save(tmp_path)
            
            executor = get_task_executor(task_id)
            future = executor.submit(upload_to_hf_with_retry, full_hf_path, tmp_path, img_name)
            batch_futures.append((future, page_num + 1))
            
            pages_processed += 1
            checkpoint_counter += 1
            
            del pix
            del page
            
            if len(batch_futures) >= BATCH_SIZE:
                for future, pg_num in batch_futures:
                    if is_task_cancelled(task_id):
                        break
                    try:
                        future.result(timeout=120)
                        mark_page_uploaded(checkpoint, pdf['name'], pg_num)
                        checkpoint['current'] = pdf['name']
                        checkpoint['last_page'] = pg_num
                    except FutureTimeoutError:
                        throttled_send(chat_id, f"⚠️ Page {pg_num} upload timeout")
                    except Exception as e:
                        print(f"Upload failed for page {pg_num}: {e}")
                batch_futures.clear()
                checkpoint_counter = 0
                
                # Async checkpoint save (non-blocking)
                async_save_checkpoint(folder_key, checkpoint)
                
                throttled_send(chat_id, f"📊 {sub_folder}: {page_num+1}/{total_pages} পৃষ্ঠা প্রসেসিংয়ে")
        
        # Process remaining batch
        for future, pg_num in batch_futures:
            if is_task_cancelled(task_id):
                break
            try:
                future.result(timeout=120)
                mark_page_uploaded(checkpoint, pdf['name'], pg_num)
                checkpoint['current'] = pdf['name']
                checkpoint['last_page'] = pg_num
            except FutureTimeoutError:
                throttled_send(chat_id, f"⚠️ Page {pg_num} upload timeout")
            except Exception as e:
                print(f"Upload failed for page {pg_num}: {e}")
        batch_futures.clear()
        
        # Final async checkpoint save
        async_save_checkpoint(folder_key, checkpoint)
        
        return pages_processed, total_pages, False
        
    finally:
        if doc:
            doc.close()
        try:
            os.unlink(pdf_path)
        except:
            pass

def process_pdfs(folder_key: str, folder_name: str, chat_id: int, task_id: str):
    """PDF প্রসেস করে (with cancel support and timeout)"""
    clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    throttled_send(chat_id, f"🚀 *প্রসেসিং শুরু হয়েছে!*\n\n📁 ফোল্ডার: `{clean_folder_name}`")
    
    overall_start_time = time.time()
    is_cancelled = False
    
    try:
        pdf_files = get_mediafire_files(folder_key)
        throttled_send(chat_id, f"📚 {len(pdf_files)}টি PDF পাওয়া গেছে।")
        
        if not pdf_files:
            throttled_send(chat_id, "❌ কোনো PDF পাওয়া যায়নি!")
            return
        
        checkpoint = load_checkpoint(folder_key)
        processed = set(checkpoint.get('processed', []))
        
        for pdf in pdf_files:
            if is_task_cancelled(task_id):
                is_cancelled = True
                throttled_send(chat_id, f"⛔ *Task cancelled by user*")
                break
            
            if time.time() - overall_start_time > MAX_JOB_RUNTIME:
                throttled_send(chat_id, f"⏰ *Job timeout* after {MAX_JOB_RUNTIME/3600} hours")
                break
            
            if pdf['name'] in processed:
                throttled_send(chat_id, f"⏭️ স্কিপ: {pdf['name']}")
                continue
            
            try:
                pages_processed, total_pages, cancelled = process_single_pdf(pdf, clean_folder_name, folder_key, chat_id, checkpoint, task_id)
                
                if cancelled:
                    is_cancelled = True
                    break
                
                processed.add(pdf['name'])
                checkpoint['processed'] = list(processed)
                checkpoint['current'] = None
                checkpoint['last_page'] = 0
                async_save_checkpoint(folder_key, checkpoint)
                
                throttled_send(chat_id, f"✅ *সম্পন্ন: {pdf['name']}*\n📄 {total_pages} পৃষ্ঠা, 🚀 {pages_processed} পৃষ্ঠা প্রসেসিত")
                
            except Exception as e:
                error_msg = traceback.format_exc()
                throttled_send(chat_id, f"❌ *Failed: {pdf['name']}*\n```\n{error_msg[:500]}\n```")
                continue
        
        with running_tasks_lock:
            if task_id in running_tasks:
                if is_cancelled:
                    running_tasks[task_id]["status"] = "cancelled"
                else:
                    running_tasks[task_id]["status"] = "completed"
                running_tasks[task_id]["completed_at"] = time.time()
        
        if not is_cancelled:
            throttled_send(chat_id, f"🎉 *সব প্রসেস সম্পন্ন!*\n\n📁 ডেটাসেট: `{HF_DATASET}/{clean_folder_name}`")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        throttled_send(chat_id, f"❌ *এরর:*\n```\n{error_msg[:1000]}\n```")
        with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error"] = str(e)[:500]
                running_tasks[task_id]["failed_at"] = time.time()
    
    finally:
        with running_tasks_lock:
            running_tasks.pop(task_id, None)
        with task_controls_lock:
            task_controls.pop(task_id, None)
        cleanup_task_executor(task_id)

# ============ টেলিগ্রাম বট হ্যান্ডলার ============
async def tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 *MediaFire to HuggingFace Processor*\n\n"
        "আমাকে একটি MediaFire ফোল্ডার লিংক দিন।\n"
        "লিংক পাঠান এখন!",
        parse_mode="Markdown"
    )

async def tg_handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.strip()
    folder_key, folder_name = extract_from_mediafire_url(url)
    if not folder_key:
        await update.message.reply_text("❌ ভুল লিংক!")
        return
    await update.message.reply_text(
        f"📁 *ফোল্ডারের তথ্য:*\n🔑 কী: `{folder_key}`\n📂 নাম: `{folder_name}`\n\n✅ প্রসেসিং শুরু করতে /confirm",
        parse_mode="Markdown"
    )
    context.user_data['folder_key'] = folder_key
    context.user_data['folder_name'] = folder_name

async def tg_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_key = context.user_data.get('folder_key')
    folder_name = context.user_data.get('folder_name')
    if not folder_key:
        await update.message.reply_text("❌ আগে একটি MediaFire লিংক দিন।")
        return
    await update.message.reply_text(f"🚀 *প্রসেসিং শুরু হচ্ছে...*\n📂 ফোল্ডার: `{folder_name}`", parse_mode="Markdown")
    
    task_id = f"{folder_key}_{update.effective_chat.id}_{int(time.time())}"
    
    with running_tasks_lock:
        running_tasks[task_id] = {
            "status": "running",
            "started_at": time.time(),
            "folder_key": folder_key,
            "folder_name": folder_name,
            "chat_id": update.effective_chat.id
        }
    
    with task_controls_lock:
        task_controls[task_id] = {"cancel": False}
    
    asyncio.create_task(
        asyncio.to_thread(
            process_pdfs, 
            folder_key, 
            folder_name, 
            update.effective_chat.id,
            task_id
        )
    )
    
    await update.message.reply_text("✅ প্রসেসিং শুরু হয়েছে!\nসমাপ্ত হলে আমি জানিয়ে দেব।\n/cancel দিয়ে বন্ধ করতে পারেন।\n/status দিয়ে অগ্রগতি দেখুন।")

async def tg_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_key = context.user_data.get('folder_key')
    if not folder_key:
        await update.message.reply_text("❌ কোনো সক্রিয় প্রসেস নেই।")
        return
    
    task_id_to_cancel = None
    with running_tasks_lock:
        for tid, task in running_tasks.items():
            if task.get("folder_key") == folder_key and task.get("status") == "running":
                task_id_to_cancel = tid
                break
    
    if task_id_to_cancel:
        with task_controls_lock:
            if task_id_to_cancel in task_controls:
                task_controls[task_id_to_cancel]["cancel"] = True
                await update.message.reply_text("⛔ *প্রসেস বন্ধ করার অনুরোধ পাঠানো হয়েছে...*\nবর্তমান কাজ শেষ করে বন্ধ হবে।", parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ টাস্ক কন্ট্রোল পাওয়া যায়নি।")
    else:
        await update.message.reply_text("❌ কোনো চলমান প্রসেস নেই।")
    
    context.user_data.clear()

async def tg_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_key = context.user_data.get('folder_key')
    if not folder_key:
        await update.message.reply_text("❌ কোনো সক্রিয় প্রসেস নেই। প্রথমে একটি MediaFire লিংক দিন।")
        return
    
    checkpoint = load_checkpoint(folder_key)
    await update.message.reply_text(
        f"📊 *বর্তমান অবস্থা:*\n✅ প্রসেস হয়েছে: {len(checkpoint.get('processed', []))}টি PDF\n"
        f"🔄 চলমান: {checkpoint.get('current', 'কিছু না')}\n"
        f"📄 শেষ পৃষ্ঠা: {checkpoint.get('last_page', 0)}",
        parse_mode="Markdown"
    )

async def setup_bot():
    """বট সেটআপ করে - সম্পূর্ণ ইনিশিয়ালাইজেশন সহ"""
    global application, bot, http_client, polling_task, main_loop
    global telegram_worker_running, telegram_worker_thread, checkpoint_writer_running
    
    main_loop = asyncio.get_running_loop()
    http_client = httpx.AsyncClient(timeout=30.0)
    
    # Start checkpoint writer worker
    if not checkpoint_writer_running:
        checkpoint_writer_running = True
        writer_thread = threading.Thread(target=checkpoint_writer_worker, daemon=True)
        writer_thread.start()
    
    # Start telegram worker thread
    if not telegram_worker_running:
        telegram_worker_running = True
        telegram_worker_thread = threading.Thread(target=telegram_worker, daemon=True)
        telegram_worker_thread.start()
    
    if TELEGRAM_BOT_TOKEN:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.initialize()
        
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        await application.initialize()
        
        application.add_handler(CommandHandler('start', tg_start))
        application.add_handler(CommandHandler('confirm', tg_confirm))
        application.add_handler(CommandHandler('cancel', tg_cancel))
        application.add_handler(CommandHandler('status', tg_status))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, tg_handle_link))
        
        polling_task = asyncio.create_task(application.run_polling())
        
        print("🤖 Telegram Bot started with safe polling.")
    else:
        print("⚠️ TELEGRAM_BOT_TOKEN not set.")

async def shutdown_bot():
    """শাটডাউনের সময় ক্লিনআপ"""
    global application, http_client, polling_task
    global telegram_worker_running, checkpoint_writer_running
    
    telegram_worker_running = False
    if telegram_worker_thread and telegram_worker_thread.is_alive():
        telegram_queue.put(None)
        telegram_worker_thread.join(timeout=5)
    
    checkpoint_writer_running = False
    checkpoint_write_queue.put(None)
    
    if polling_task:
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass
    
    if application:
        await application.shutdown()
    
    if http_client:
        await http_client.aclose()
    
    with task_executors_lock:
        for executor in task_executors.values():
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except:
                pass
        task_executors.clear()

# ============ FastAPI অ্যাপ ============
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await setup_bot()

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_bot()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Tafsir Image Processor is running"}

@app.post("/start_processing")
async def start_processing(request: ProcessRequest):
    if not request.folder_key or not request.folder_name:
        raise HTTPException(status_code=400, detail="folder_key and folder_name required")
    
    task_id = f"{request.folder_key}_{request.telegram_chat_id}_{int(time.time())}"
    
    with running_tasks_lock:
        running_tasks[task_id] = {
            "status": "running",
            "started_at": time.time(),
            "folder_key": request.folder_key,
            "folder_name": request.folder_name,
            "chat_id": request.telegram_chat_id
        }
    
    with task_controls_lock:
        task_controls[task_id] = {"cancel": False}
    
    asyncio.create_task(
        asyncio.to_thread(
            process_pdfs, 
            request.folder_key, 
            request.folder_name, 
            request.telegram_chat_id,
            task_id
        )
    )
    
    return {"status": "started", "message": "Processing started in background", "task_id": task_id}

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    with task_controls_lock:
        if task_id in task_controls:
            task_controls[task_id]["cancel"] = True
            return {"status": "cancelling", "message": "Task cancellation requested"}
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/status/{folder_key}")
def get_status(folder_key: str):
    checkpoint = load_checkpoint(folder_key)
    return {
        "processed": len(checkpoint.get('processed', [])),
        "current": checkpoint.get('current'),
        "last_page": checkpoint.get('last_page', 0)
    }

@app.get("/tasks")
def get_tasks():
    with running_tasks_lock:
        return {k: v for k, v in running_tasks.items()}

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)