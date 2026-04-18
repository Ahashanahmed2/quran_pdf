#!/usr/bin/env python3
"""
একক ফাইল: FastAPI + Telegram Bot (Internet Archive + MediaFire Support)
উভয় পদ্ধতি সমর্থন করে:
1. আইটেম লিংক: https://archive.org/details/ITEM_ID
2. সরাসরি PDF লিংক: https://archive.org/download/ITEM_ID/1.pdf
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
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import Update, Bot
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

CHECKPOINT_DIR = Path("/tmp/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MIN_FREE_SPACE_MB = 500
UPLOAD_CONCURRENCY = 3
MAX_PENDING_UPLOADS = 5
TEMP_FILE_MAX_AGE_HOURS = 24
PAGE_HASH_CACHE_SIZE = 10000
MAX_JOB_RUNTIME = 6 * 3600
MEDIAFIRE_API_DELAY = 0.5
# =====================================

# গ্লোবাল ভেরিয়েবল
application = None
mediafire_session = None
mediafire_session_lock = asyncio.Lock()
_bot_initialized = False
_bot_init_lock = asyncio.Lock()
_updater_started = False

running_tasks = {}
running_tasks_lock = asyncio.Lock()
task_controls = {}
task_controls_lock = asyncio.Lock()

upload_semaphore = asyncio.Semaphore(UPLOAD_CONCURRENCY)

last_telegram_sent = {}
telegram_lock = asyncio.Lock()
telegram_queue = asyncio.Queue(maxsize=2000)
telegram_worker_task = None
telegram_dropped_count = 0
telegram_dropped_lock = asyncio.Lock()

page_hash_cache = OrderedDict()
page_hash_cache_lock = asyncio.Lock()

checkpoint_write_queue = asyncio.Queue()
checkpoint_writer_task = None
checkpoint_lock = asyncio.Lock()

disk_cleanup_task = None
shutdown_in_progress = False

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens=1):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            wait_time = (tokens - self.tokens) / self.rate
            self.last_refill = now
            await asyncio.sleep(wait_time)
            self.tokens = 0
            return True

mediafire_rate_limiter = TokenBucket(rate=1.0, capacity=2.0)

class ProcessRequest(BaseModel):
    folder_key: str = None
    folder_name: str = None
    telegram_chat_id: int
    pdf_urls: list = None

# ============ Disk Space Check ============
async def check_disk_space():
    try:
        total, used, free = shutil.disk_usage("/tmp")
        free_mb = free // (1024 * 1024)
        if free_mb < MIN_FREE_SPACE_MB:
            raise RuntimeError(f"Low disk space: {free_mb} MB free")
        return True
    except RuntimeError:
        raise
    except Exception as e:
        print(f"[Disk Check] Warning: {e}")
        return True

# ============ Background Disk Cleanup ============
async def disk_cleanup_worker():
    while not shutdown_in_progress:
        await asyncio.sleep(1800)
        try:
            now = time.time()
            for f in TEMP_DIR.glob("*"):
                try:
                    if f.is_file() and now - f.stat().st_mtime > TEMP_FILE_MAX_AGE_HOURS * 3600:
                        f.unlink()
                except:
                    pass
            print("[Cleanup] Temp files cleaned", flush=True)
        except Exception as e:
            print(f"[Cleanup] Error: {e}", flush=True)

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

# ============ Checkpoint Async Writer ============
def get_checkpoint_path(folder_key):
    safe_key = folder_key.replace('/', '_').replace('\\', '_')
    return CHECKPOINT_DIR / f"checkpoint_{safe_key}.json"

async def checkpoint_writer_worker():
    while not shutdown_in_progress:
        try:
            item = await asyncio.wait_for(checkpoint_write_queue.get(), timeout=1.0)
            if item is None:
                break
            folder_key, checkpoint = item
            async with checkpoint_lock:
                checkpoint_path = get_checkpoint_path(folder_key)
                temp_path = checkpoint_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                os.replace(temp_path, checkpoint_path)
            checkpoint_write_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"[Checkpoint Writer] Error: {e}", flush=True)

async def async_save_checkpoint(folder_key, checkpoint):
    try:
        await asyncio.wait_for(checkpoint_write_queue.put((folder_key, checkpoint.copy())), timeout=1.0)
    except asyncio.TimeoutError:
        print(f"[Checkpoint] Queue full, sync saving {folder_key}", flush=True)
        async with checkpoint_lock:
            checkpoint_path = get_checkpoint_path(folder_key)
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            os.replace(temp_path, checkpoint_path)

async def load_checkpoint(folder_key):
    async with checkpoint_lock:
        checkpoint_path = get_checkpoint_path(folder_key)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[Checkpoint] Corrupted, resetting", flush=True)
                return {"processed": [], "current": None, "last_page": 0, "last_page_map": {}}
    return {"processed": [], "current": None, "last_page": 0, "last_page_map": {}}

# ============ Telegram Queue Worker ============
async def send_telegram_safe(chat_id, message):
    if not TELEGRAM_BOT_TOKEN:
        return False
    for retry in range(3):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": chat_id, "text": message[:4000], "parse_mode": None}
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return True
            elif response.status_code == 429:
                wait_time = 2 ** retry
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(1)
        except Exception as e:
            await asyncio.sleep(2 ** retry)
    return False

def is_high_priority_message(message):
    high_priority_keywords = ["✅", "🎉", "❌", "🚀", "⛔", "⚠️", "সম্পন্ন", "সমাপ্ত", "error", "failed"]
    return any(keyword in message for keyword in high_priority_keywords)

async def telegram_worker():
    global telegram_dropped_count
    while not shutdown_in_progress:
        try:
            item = await asyncio.wait_for(telegram_queue.get(), timeout=1.0)
            if item is None:
                break
            chat_id, message = item
            success = await send_telegram_safe(chat_id, message)
            if not success:
                print(f"[Telegram] Failed to send message after retries", flush=True)
            telegram_queue.task_done()
            await asyncio.sleep(0.05)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"[Telegram Worker] Error: {e}", flush=True)
            await asyncio.sleep(0.5)

async def send_telegram(chat_id, message):
    global telegram_dropped_count
    try:
        await asyncio.wait_for(telegram_queue.put((chat_id, message)), timeout=1.0)
    except asyncio.TimeoutError:
        if is_high_priority_message(message):
            await send_telegram_safe(chat_id, message)
        else:
            async with telegram_dropped_lock:
                telegram_dropped_count += 1
    except Exception as e:
        print(f"[Telegram] Queue error: {e}", flush=True)

async def throttled_send(chat_id, message, interval=2):
    async with telegram_lock:
        now = time.time()
        last_time = last_telegram_sent.get(chat_id, 0)
        if now - last_time >= interval:
            await send_telegram(chat_id, message)
            last_telegram_sent[chat_id] = now
            return True
        return False

# ============ Internet Archive: আইটেম থেকে PDF লিংক বের করা ============
async def extract_pdfs_from_archive_item(item_url: str, max_pdfs: int = 30):
    """Internet Archive আইটেম পেজ থেকে সব PDF লিংক বের করুন"""
    pdf_urls = []
    
    try:
        item_match = re.search(r'/details/([^/?]+)', item_url)
        if not item_match:
            return pdf_urls
        
        item_id = item_match.group(1)
        print(f"[Archive] Extracting from item: {item_id}", flush=True)
        
        metadata_url = f"https://archive.org/metadata/{item_id}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(metadata_url)
            if response.status_code == 200:
                data = response.json()
                files = data.get('files', [])
                for file_info in files:
                    file_name = file_info.get('name', '')
                    if file_name.lower().endswith('.pdf'):
                        pdf_url = f"https://archive.org/download/{item_id}/{file_name}"
                        pdf_urls.append(pdf_url)
                
                if not pdf_urls:
                    for i in range(1, max_pdfs + 1):
                        test_url = f"https://archive.org/download/{item_id}/{i}.pdf"
                        try:
                            test_response = await client.head(test_url)
                            if test_response.status_code == 200:
                                pdf_urls.append(test_url)
                        except:
                            pass
    except Exception as e:
        print(f"[Archive] Error extracting PDFs: {e}", flush=True)
        traceback.print_exc()
    
    return pdf_urls

# ============ PDF ডাউনলোড ও প্রসেসিং ============
async def download_pdf_stream(url):
    print(f"[DEBUG] download_pdf_stream: {url[:80]}...", flush=True)
    await check_disk_space()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
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
        traceback.print_exc()
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
            print(f"Upload retry {retry+1}/3: {e}, waiting {wait_time}s", flush=True)
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

async def process_single_pdf(pdf, clean_folder_name, folder_key, chat_id, checkpoint, task_id):
    sub_folder = str(pdf['number'])
    full_hf_path = f"{clean_folder_name}/{sub_folder}"
    await throttled_send(chat_id, f"📄 প্রসেসিং: {pdf['name']}\n📁 লোকেশন: {full_hf_path}")
    
    start_page = 0
    if checkpoint.get('current') == pdf['name']:
        start_page = checkpoint.get('last_page', 0)
        if start_page > 0:
            await throttled_send(chat_id, f"📌 Resuming from page {start_page + 1}")
    
    if not pdf.get('download_link'):
        await throttled_send(chat_id, f"❌ ডাউনলোড লিংক পাওয়া যায়নি: {pdf['name']}")
        return 0, 0, False
    
    await throttled_send(chat_id, f"⬇️ ডাউনলোড হচ্ছে: {pdf['name']}")
    pdf_path = await download_pdf_stream(pdf['download_link'])
    
    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        await throttled_send(chat_id, f"🖼️ {total_pages} পৃষ্ঠা কনভার্ট ও আপলোড শুরু...")
        
        BATCH_SIZE = 3
        batch_tasks = []
        pages_processed = 0
        job_start_time = time.time()
        
        for page_num in range(start_page, total_pages):
            if shutdown_in_progress or await is_task_cancelled(task_id):
                await throttled_send(chat_id, f"⛔ Task cancelled at page {page_num + 1}")
                return pages_processed, total_pages, True
            
            if time.time() - job_start_time > MAX_JOB_RUNTIME:
                await throttled_send(chat_id, f"⏰ Job timeout")
                return pages_processed, total_pages, False
            
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
            tmp_path = TEMP_DIR / f"{task_id}_{page_num}_{int(time.time()*1000)}.png"
            
            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)
            
            task = asyncio.create_task(upload_to_hf_with_retry(full_hf_path, tmp_path, img_name, page_hash))
            batch_tasks.append((task, page_num + 1))
            pages_processed += 1
            
            while len(batch_tasks) >= MAX_PENDING_UPLOADS:
                await asyncio.sleep(0.05)
            
            if len(batch_tasks) >= BATCH_SIZE:
                for t, pg_num in batch_tasks:
                    try:
                        await asyncio.wait_for(t, timeout=120)
                        mark_page_uploaded_checkpoint(checkpoint, pdf['name'], pg_num)
                        checkpoint['current'] = pdf['name']
                        checkpoint['last_page'] = pg_num
                    except:
                        pass
                batch_tasks.clear()
                await async_save_checkpoint(folder_key, checkpoint)
                
                if page_num % 10 == 0:
                    await throttled_send(chat_id, f"📊 {sub_folder}: {page_num+1}/{total_pages} পৃষ্ঠা")
                gc.collect()
        
        for t, pg_num in batch_tasks:
            try:
                await asyncio.wait_for(t, timeout=120)
                mark_page_uploaded_checkpoint(checkpoint, pdf['name'], pg_num)
            except:
                pass
        
        await async_save_checkpoint(folder_key, checkpoint)
        return pages_processed, total_pages, False
    finally:
        if doc:
            doc.close()
        try:
            os.unlink(pdf_path)
        except:
            pass

async def process_pdf_urls(pdf_urls: list, folder_name: str, chat_id: int, task_id: str):
    print(f"[DEBUG] ========== PROCESS STARTED ==========", flush=True)
    print(f"[DEBUG] Folder: {folder_name}", flush=True)
    print(f"[DEBUG] PDF Count: {len(pdf_urls)}", flush=True)
    print(f"[DEBUG] First URL: {pdf_urls[0][:80] if pdf_urls else 'None'}", flush=True)
    print(f"[DEBUG] Chat ID: {chat_id}", flush=True)
    print(f"[DEBUG] Task ID: {task_id}", flush=True)
    
    try:
        clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
        await throttled_send(chat_id, f"🚀 প্রসেসিং শুরু!\n📁 {clean_folder_name}\n📚 {len(pdf_urls)}টি PDF")
        
        overall_start_time = time.time()
        is_cancelled = False
        processed = []
        
        checkpoint = await load_checkpoint(f"direct_{clean_folder_name}")
        already_processed = set(checkpoint.get('processed', []))
        
        for idx, url in enumerate(pdf_urls):
            if shutdown_in_progress or await is_task_cancelled(task_id):
                is_cancelled = True
                break
            if time.time() - overall_start_time > MAX_JOB_RUNTIME:
                break
            
            name_match = re.search(r'/(\d+)\.pdf', url)
            if name_match:
                pdf_number = int(name_match.group(1))
                pdf_name = f"{pdf_number}.pdf"
            else:
                pdf_number = idx + 1
                pdf_name = url.split('/')[-1]
                if not pdf_name.endswith('.pdf'):
                    pdf_name = f"file_{idx+1}.pdf"
            
            if pdf_name in already_processed:
                await throttled_send(chat_id, f"⏭️ স্কিপ: {pdf_name}")
                continue
            
            pdf = {'name': pdf_name, 'number': pdf_number, 'download_link': url}
            await throttled_send(chat_id, f"📖 [{idx+1}/{len(pdf_urls)}] {pdf_name}")
            
            try:
                pages_processed, total_pages, cancelled = await process_single_pdf(
                    pdf, clean_folder_name, f"direct_{clean_folder_name}", chat_id, checkpoint, task_id
                )
                if cancelled:
                    is_cancelled = True
                    break
                
                processed.append(pdf_name)
                checkpoint['processed'] = list(set(checkpoint.get('processed', []) + [pdf_name]))
                await async_save_checkpoint(f"direct_{clean_folder_name}", checkpoint)
                await throttled_send(chat_id, f"✅ সম্পন্ন: {pdf_name} ({total_pages} পৃষ্ঠা)")
            except Exception as e:
                print(f"[DEBUG] Error processing {pdf_name}: {e}", flush=True)
                traceback.print_exc()
                await throttled_send(chat_id, f"❌ ব্যর্থ: {pdf_name}\n{str(e)[:200]}")
        
        if not is_cancelled and processed:
            await throttled_send(chat_id, f"🎉 সব সম্পন্ন!\n📁 {HF_DATASET}/{clean_folder_name}\n✅ {len(processed)}টি PDF")
        
        async with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "completed" if not is_cancelled else "cancelled"
    except Exception as e:
        print(f"[DEBUG] FATAL ERROR in process_pdf_urls: {e}", flush=True)
        traceback.print_exc()
        await throttled_send(chat_id, f"❌ সিস্টেম ত্রুটি: {str(e)[:300]}")

# ============ টেলিগ্রাম বট হ্যান্ডলার ============
async def tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 PDF to HuggingFace Processor\n\n"
        "সাপোর্টেড লিংক:\n"
        "✅ Internet Archive আইটেম: https://archive.org/details/...\n"
        "✅ Internet Archive PDF: https://archive.org/download/.../1.pdf\n"
        "✅ MediaFire PDF: https://www.mediafire.com/file/.../1.pdf/file\n\n"
        "ব্যবহার:\n"
        "প্রথম লাইনে বইয়ের নাম\n"
        "তারপর লিংক\n"
        "তারপর /confirm\n\n"
        "কমান্ড:\n"
        "/start, /bookname, /confirm, /status, /cancel, /debug"
    )

async def tg_bookname(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ ব্যবহার: /bookname বইয়ের_নাম")
        return
    book_name = ' '.join(context.args)
    context.user_data['folder_name'] = book_name
    await update.message.reply_text(f"✅ বই: {book_name}")

async def tg_handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[DEBUG] tg_handle_link called", flush=True)
    text = update.message.text.strip()
    lines = text.split('\n')
    
    pdf_urls = []
    book_name = None
    archive_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if 'archive.org/details/' in line:
            archive_item = line
        elif ('archive.org/download/' in line or '/file/' in line) and '.pdf' in line:
            pdf_urls.append(line)
        elif not book_name and not line.startswith('http'):
            book_name = line
    
    if archive_item and not pdf_urls:
        await update.message.reply_text("🔍 Internet Archive থেকে PDF লিংক বের করা হচ্ছে...")
        pdf_urls = await extract_pdfs_from_archive_item(archive_item)
        
        if pdf_urls:
            def extract_number(url):
                match = re.search(r'/(\d+)\.pdf', url)
                return int(match.group(1)) if match else 999
            pdf_urls.sort(key=extract_number)
            await update.message.reply_text(f"✅ {len(pdf_urls)}টি PDF লিংক পাওয়া গেছে!")
        else:
            await update.message.reply_text("❌ কোনো PDF লিংক বের করা যায়নি!")
            return
    
    if pdf_urls:
        context.user_data['pdf_urls'] = pdf_urls
        if book_name:
            context.user_data['folder_name'] = book_name
        elif not context.user_data.get('folder_name'):
            context.user_data['folder_name'] = f"tafsir_{int(time.time())}"
        
        preview = "\n".join(pdf_urls[:5])
        if len(pdf_urls) > 5:
            preview += f"\n... এবং আরো {len(pdf_urls) - 5}টি"
        
        await update.message.reply_text(
            f"📚 বই: {context.user_data['folder_name']}\n"
            f"📄 {len(pdf_urls)}টি PDF\n\n"
            f"লিংকসমূহ:\n{preview}\n\n"
            f"✅ /confirm"
        )
        return
    
    await update.message.reply_text("❌ কোনো PDF লিংক পাওয়া যায়নি!")

async def tg_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[DEBUG] tg_confirm called", flush=True)
    print(f"[DEBUG] user_data keys: {list(context.user_data.keys())}", flush=True)
    
    pdf_urls = context.user_data.get('pdf_urls')
    folder_name = context.user_data.get('folder_name')
    
    print(f"[DEBUG] pdf_urls count: {len(pdf_urls) if pdf_urls else 0}", flush=True)
    print(f"[DEBUG] folder_name: {folder_name}", flush=True)
    
    if pdf_urls:
        if not folder_name:
            folder_name = f"tafsir_{int(time.time())}"
        
        await update.message.reply_text(f"🚀 শুরু হচ্ছে...\n📚 {len(pdf_urls)} PDF\n📁 {folder_name}")
        
        task_id = f"direct_{update.effective_chat.id}_{int(time.time())}"
        async with running_tasks_lock:
            running_tasks[task_id] = {"status": "running", "started_at": time.time()}
        async with task_controls_lock:
            task_controls[task_id] = {"cancel": False}
        
        asyncio.create_task(process_pdf_urls(pdf_urls, folder_name, update.effective_chat.id, task_id))
        await update.message.reply_text("✅ শুরু হয়েছে! /status দেখুন")
        context.user_data['pdf_urls'] = None
        return
    
    await update.message.reply_text("❌ আগে PDF লিংক দিন")

async def tg_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    task_to_cancel = None
    async with running_tasks_lock:
        for tid, task in running_tasks.items():
            if task.get("status") == "running":
                task_to_cancel = tid
                break
    
    if task_to_cancel:
        async with task_controls_lock:
            if task_to_cancel in task_controls:
                task_controls[task_to_cancel]["cancel"] = True
        await update.message.reply_text("⛔ বন্ধ করা হচ্ছে...")
    else:
        await update.message.reply_text("❌ কোনো চলমান প্রসেস নেই")

async def tg_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_name = context.user_data.get('folder_name', 'tafsir')
    clean_name = folder_name.replace(' ', '_')
    checkpoint = await load_checkpoint(f"direct_{clean_name}")
    
    processed = checkpoint.get('processed', [])
    current = checkpoint.get('current', 'কিছু না')
    last_page = checkpoint.get('last_page', 0)
    
    await update.message.reply_text(
        f"📊 অবস্থা:\n"
        f"✅ প্রসেস: {len(processed)}টি PDF\n"
        f"🔄 চলমান: {current}\n"
        f"📄 শেষ পৃষ্ঠা: {last_page}"
    )

async def tg_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = [
        "🔧 স্ট্যাটাস:",
        f"📧 MediaFire: {'✅' if MEDIAFIRE_EMAIL else '❌'}",
        f"🤗 HF Token: {'✅' if HF_TOKEN else '❌'}",
        f"📁 HF Dataset: {HF_DATASET or '❌'}",
        f"🤖 Bot Token: {'✅' if TELEGRAM_BOT_TOKEN else '❌'}",
        f"⚙️ চলমান টাস্ক: {len(running_tasks)}"
    ]
    try:
        total, used, free = shutil.disk_usage("/tmp")
        status.append(f"💾 ফ্রি: {free // (1024*1024)} MB")
    except:
        pass
    await update.message.reply_text("\n".join(status))

# ============ Telegram Bot Setup ============
async def setup_bot():
    global application, telegram_worker_task, checkpoint_writer_task, disk_cleanup_task
    global _bot_initialized, _updater_started
    
    async with _bot_init_lock:
        if _bot_initialized:
            return
        print("🔧 setup_bot() শুরু", flush=True)
        
        telegram_worker_task = asyncio.create_task(telegram_worker())
        checkpoint_writer_task = asyncio.create_task(checkpoint_writer_worker())
        disk_cleanup_task = asyncio.create_task(disk_cleanup_worker())
        
        if not TELEGRAM_BOT_TOKEN:
            print("⚠️ TELEGRAM_BOT_TOKEN নেই", flush=True)
            return
        
        try:
            temp_bot = Bot(token=TELEGRAM_BOT_TOKEN)
            await temp_bot.delete_webhook(drop_pending_updates=True)
            await asyncio.sleep(2)
            await temp_bot.close()
            print("✅ সেশন ক্লিয়ার", flush=True)
        except Exception as e:
            print(f"⚠️ সেশন ক্লিয়ার সমস্যা: {e}", flush=True)
        
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        application.add_handler(CommandHandler('start', tg_start))
        application.add_handler(CommandHandler('bookname', tg_bookname))
        application.add_handler(CommandHandler('confirm', tg_confirm))
        application.add_handler(CommandHandler('cancel', tg_cancel))
        application.add_handler(CommandHandler('status', tg_status))
        application.add_handler(CommandHandler('debug', tg_debug))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, tg_handle_link))
        
        await application.initialize()
        await application.start()
        
        if not _updater_started:
            asyncio.create_task(application.updater.start_polling(poll_interval=1.0, timeout=30, drop_pending_updates=True))
            _updater_started = True
            print("🤖 বট চলছে", flush=True)
        
        _bot_initialized = True

async def shutdown_bot():
    global shutdown_in_progress, application
    shutdown_in_progress = True
    print("🛑 বন্ধ হচ্ছে...", flush=True)
    if application:
        await application.stop()
        await application.shutdown()

# ============ FastAPI অ্যাপ ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_bot()
    yield
    await shutdown_bot()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Tafsir Image Processor"}

@app.get("/tasks")
async def get_tasks():
    async with running_tasks_lock:
        return {k: v for k, v in running_tasks.items()}

@app.get("/test-download")
async def test_download():
    """PDF ডাউনলোড টেস্ট করুন"""
    try:
        url = "https://archive.org/download/20260415_20260415_0945/1.pdf"
        
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.head(url)
            
        return {
            "status": "success",
            "url": url,
            "response_code": response.status_code,
            "headers": dict(response.headers)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-archive")
async def test_archive():
    """Internet Archive সংযোগ টেস্ট"""
    try:
        url = "https://archive.org"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
        return {"status": "success", "archive.org": response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)