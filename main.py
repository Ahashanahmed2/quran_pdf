#!/usr/bin/env python3
"""
একক ফাইল: FastAPI + Telegram Bot (Production Ready - Fixed Event Loop Conflict)
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
from telegram import Update
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

# Minimum free disk space (500 MB)
MIN_FREE_SPACE_MB = 500

# Upload concurrency
UPLOAD_CONCURRENCY = 3
MAX_PENDING_UPLOADS = 5

# Temp file cleanup age (hours)
TEMP_FILE_MAX_AGE_HOURS = 24

# Page hash cache size
PAGE_HASH_CACHE_SIZE = 10000

# Job timeout (6 hours)
MAX_JOB_RUNTIME = 6 * 3600

# MediaFire API delay
MEDIAFIRE_API_DELAY = 0.5
# =====================================

# গ্লোবাল ভেরিয়েবল
application = None
mediafire_session = None
mediafire_session_lock = asyncio.Lock()

# Bot initialization tracking
_bot_initialized = False
_bot_init_lock = asyncio.Lock()
_updater_started = False

# Task management
running_tasks = {}
running_tasks_lock = asyncio.Lock()
task_controls = {}
task_controls_lock = asyncio.Lock()

# Rate limit control
upload_semaphore = asyncio.Semaphore(UPLOAD_CONCURRENCY)

# Telegram throttling
last_telegram_sent = {}
telegram_lock = asyncio.Lock()
telegram_queue = asyncio.Queue(maxsize=2000)
telegram_worker_task = None
telegram_dropped_count = 0
telegram_dropped_lock = asyncio.Lock()

# Page hash cache (LRU using OrderedDict)
page_hash_cache = OrderedDict()
page_hash_cache_lock = asyncio.Lock()

# Checkpoint async writer
checkpoint_write_queue = asyncio.Queue()
checkpoint_writer_task = None
checkpoint_lock = asyncio.Lock()

# Disk cleanup task
disk_cleanup_task = None

# Shutdown flag
shutdown_in_progress = False

# MediaFire rate limiter
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
    folder_key: str
    folder_name: str
    telegram_chat_id: int

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
    """Periodic cleanup of old temp files"""
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
            print("[Cleanup] Temp files cleaned")
        except Exception as e:
            print(f"[Cleanup] Error: {e}")

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
            print(f"[Checkpoint Writer] Error: {e}")

async def async_save_checkpoint(folder_key, checkpoint):
    try:
        await asyncio.wait_for(checkpoint_write_queue.put((folder_key, checkpoint.copy())), timeout=1.0)
    except asyncio.TimeoutError:
        print(f"[Checkpoint] Queue full, sync saving {folder_key}")
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
                print(f"[Checkpoint] Corrupted, resetting")
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
                print(f"[Telegram] Rate limited, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                print(f"[Telegram] HTTP {response.status_code}")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"[Telegram] Send error (retry {retry+1}/3): {e}")
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
                print(f"[Telegram] Failed to send message after retries")

            telegram_queue.task_done()
            await asyncio.sleep(0.05)

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"[Telegram Worker] Error: {e}")
            await asyncio.sleep(0.5)

async def send_telegram(chat_id, message):
    global telegram_dropped_count

    try:
        await asyncio.wait_for(telegram_queue.put((chat_id, message)), timeout=1.0)
    except asyncio.TimeoutError:
        if is_high_priority_message(message):
            await send_telegram_safe(chat_id, message)
            print(f"[Telegram] High-priority message sent directly")
        else:
            async with telegram_dropped_lock:
                telegram_dropped_count += 1
                if telegram_dropped_count % 20 == 0:
                    print(f"[Telegram] {telegram_dropped_count} low-priority messages dropped")
    except Exception as e:
        print(f"[Telegram] Queue error: {e}")

async def throttled_send(chat_id, message, interval=2):
    async with telegram_lock:
        now = time.time()
        last_time = last_telegram_sent.get(chat_id, 0)
        if now - last_time >= interval:
            await send_telegram(chat_id, message)
            last_telegram_sent[chat_id] = now
            return True
        return False

# ============ MediaFire Session Management ============
async def get_mediafire_session():
    global mediafire_session
    async with mediafire_session_lock:
        if mediafire_session is None:
            try:
                api = MediaFireApi()
                session = api.user_get_session_token(
                    email=MEDIAFIRE_EMAIL, 
                    password=MEDIAFIRE_PASSWORD, 
                    app_id='42511'
                )
                api.session = session
                mediafire_session = api
                print("✅ MediaFire session created")
            except Exception as e:
                print(f"⚠️ Failed to create MediaFire session: {e}")
                raise
        return mediafire_session

# ============ PDF প্রসেসিং ফাংশন ============
def extract_from_mediafire_url(url):
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

async def download_pdf_stream(url):
    await check_disk_space()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=8192):
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
            return True
        except Exception as e:
            if retry == 2:
                raise
            wait_time = 2 ** retry
            print(f"Upload retry {retry+1}/3: {e}, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
    return False

async def get_mediafire_files(folder_key):
    await mediafire_rate_limiter.acquire()

    api = await get_mediafire_session()

    def sync_call():
        return api.folder_get_content(folder_key=folder_key)

    folder_content = await asyncio.to_thread(sync_call)

    files = []
    for item in folder_content['folder_content']:
        if item['type'] == 'file' and item['filename'].endswith('.pdf'):
            def sync_file_links():
                return api.file_get_links(quickkey=item['quickkey'])

            file_links = await asyncio.to_thread(sync_file_links)
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
            await asyncio.sleep(MEDIAFIRE_API_DELAY)
    files.sort(key=lambda x: x['number'])
    return files

async def process_single_pdf(pdf, clean_folder_name, folder_key, chat_id, checkpoint, task_id):
    sub_folder = str(pdf['number'])
    full_hf_path = f"{clean_folder_name}/{sub_folder}"

    await throttled_send(chat_id, f"📄 প্রসেসিং: {pdf['name']}\n📁 লোকেশন: {full_hf_path}")

    start_page = 0
    if checkpoint.get('current') == pdf['name']:
        start_page = checkpoint.get('last_page', 0)
        if start_page > 0:
            await throttled_send(chat_id, f"📌 Resuming from page {start_page + 1}")

    pdf_path = await download_pdf_stream(pdf['download_link'])

    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        await throttled_send(chat_id, f"🖼️ {total_pages} পৃষ্ঠা কনভার্ট ও আপলোড শুরু...")

        BATCH_SIZE = 3
        batch_tasks = []
        pages_processed = 0
        checkpoint_counter = 0
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
                await throttled_send(chat_id, f"⏭️ Page {page_num + 1} already exists (duplicate), skipping")
                mark_page_uploaded_checkpoint(checkpoint, pdf['name'], page_num + 1)
                pix = None
                page = None
                continue

            img_name = f"page_{page_num+1:04d}.png"
            tmp_path = TEMP_DIR / f"{task_id}_{page_num}_{int(time.time()*1000)}.png"

            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)

            pix = None
            page = None

            task = asyncio.create_task(upload_to_hf_with_retry(full_hf_path, tmp_path, img_name, page_hash))
            batch_tasks.append((task, page_num + 1, page_hash))
            pages_processed += 1
            checkpoint_counter += 1

            while len(batch_tasks) >= MAX_PENDING_UPLOADS:
                await asyncio.sleep(0.05)
                if shutdown_in_progress or await is_task_cancelled(task_id):
                    return pages_processed, total_pages, True

            if len(batch_tasks) >= BATCH_SIZE:
                for task, pg_num, pg_hash in batch_tasks:
                    if shutdown_in_progress or await is_task_cancelled(task_id):
                        break
                    try:
                        await asyncio.wait_for(task, timeout=120)
                        mark_page_uploaded_checkpoint(checkpoint, pdf['name'], pg_num)
                        checkpoint['current'] = pdf['name']
                        checkpoint['last_page'] = pg_num
                    except asyncio.TimeoutError:
                        await throttled_send(chat_id, f"⚠️ Page {pg_num} upload timeout")
                    except Exception as e:
                        print(f"Upload failed for page {pg_num}: {e}")
                batch_tasks.clear()
                checkpoint_counter = 0
                await async_save_checkpoint(folder_key, checkpoint)
                await throttled_send(chat_id, f"📊 {sub_folder}: {page_num+1}/{total_pages} পৃষ্ঠা প্রসেসিংয়ে")

                gc.collect()

        for task, pg_num, pg_hash in batch_tasks:
            if shutdown_in_progress or await is_task_cancelled(task_id):
                break
            try:
                await asyncio.wait_for(task, timeout=120)
                mark_page_uploaded_checkpoint(checkpoint, pdf['name'], pg_num)
                checkpoint['current'] = pdf['name']
                checkpoint['last_page'] = pg_num
            except asyncio.TimeoutError:
                await throttled_send(chat_id, f"⚠️ Page {pg_num} upload timeout")
            except Exception as e:
                print(f"Upload failed for page {pg_num}: {e}")
        batch_tasks.clear()

        await async_save_checkpoint(folder_key, checkpoint)
        gc.collect()

        return pages_processed, total_pages, False

    finally:
        if doc:
            doc.close()
        try:
            os.unlink(pdf_path)
        except:
            pass

async def process_pdfs(folder_key: str, folder_name: str, chat_id: int, task_id: str):
    clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    await throttled_send(chat_id, f"🚀 প্রসেসিং শুরু হয়েছে!\n\n📁 ফোল্ডার: {clean_folder_name}")

    overall_start_time = time.time()
    is_cancelled = False

    try:
        pdf_files = await get_mediafire_files(folder_key)
        await throttled_send(chat_id, f"📚 {len(pdf_files)}টি PDF পাওয়া গেছে।")

        if not pdf_files:
            await throttled_send(chat_id, "❌ কোনো PDF পাওয়া যায়নি!")
            return

        checkpoint = await load_checkpoint(folder_key)
        processed = set(checkpoint.get('processed', []))

        for pdf in pdf_files:
            if shutdown_in_progress or await is_task_cancelled(task_id):
                is_cancelled = True
                await throttled_send(chat_id, f"⛔ Task cancelled")
                break

            if time.time() - overall_start_time > MAX_JOB_RUNTIME:
                await throttled_send(chat_id, f"⏰ Job timeout")
                break

            if pdf['name'] in processed:
                await throttled_send(chat_id, f"⏭️ স্কিপ: {pdf['name']}")
                continue

            try:
                pages_processed, total_pages, cancelled = await process_single_pdf(pdf, clean_folder_name, folder_key, chat_id, checkpoint, task_id)

                if cancelled:
                    is_cancelled = True
                    break

                processed.add(pdf['name'])
                checkpoint['processed'] = list(processed)
                checkpoint['current'] = None
                checkpoint['last_page'] = 0
                await async_save_checkpoint(folder_key, checkpoint)

                await throttled_send(chat_id, f"✅ সম্পন্ন: {pdf['name']}\n📄 {total_pages} পৃষ্ঠা, 🚀 {pages_processed} পৃষ্ঠা প্রসেসিত")

            except Exception as e:
                error_msg = traceback.format_exc()
                await throttled_send(chat_id, f"❌ Failed: {pdf['name']}\n{error_msg[:300]}")
                continue

        async with running_tasks_lock:
            if task_id in running_tasks:
                if is_cancelled:
                    running_tasks[task_id]["status"] = "cancelled"
                else:
                    running_tasks[task_id]["status"] = "completed"
                running_tasks[task_id]["completed_at"] = time.time()

        if not is_cancelled:
            await throttled_send(chat_id, f"🎉 সব প্রসেস সম্পন্ন!\n\n📁 ডেটাসেট: {HF_DATASET}/{clean_folder_name}")

    except Exception as e:
        error_msg = traceback.format_exc()
        await throttled_send(chat_id, f"❌ Error:\n{error_msg[:500]}")
        async with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error"] = str(e)[:500]
                running_tasks[task_id]["failed_at"] = time.time()

    finally:
        async with running_tasks_lock:
            running_tasks.pop(task_id, None)
        async with task_controls_lock:
            task_controls.pop(task_id, None)

# ============ টেলিগ্রাম বট হ্যান্ডলার ============
async def tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 MediaFire to HuggingFace Processor\n\n"
        "আমাকে একটি MediaFire ফোল্ডার লিংক দিন।\n"
        "লিংক পাঠান এখন!",
        parse_mode=None
    )

async def tg_handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.strip()
    folder_key, folder_name = extract_from_mediafire_url(url)
    if not folder_key:
        await update.message.reply_text("❌ ভুল লিংক!")
        return
    await update.message.reply_text(
        f"📁 ফোল্ডারের তথ্য:\n🔑 কী: {folder_key}\n📂 নাম: {folder_name}\n\n✅ প্রসেসিং শুরু করতে /confirm",
        parse_mode=None
    )
    context.user_data['folder_key'] = folder_key
    context.user_data['folder_name'] = folder_name

async def tg_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_key = context.user_data.get('folder_key')
    folder_name = context.user_data.get('folder_name')
    if not folder_key:
        await update.message.reply_text("❌ আগে একটি MediaFire লিংক দিন।")
        return
    await update.message.reply_text(f"🚀 প্রসেসিং শুরু হচ্ছে...\n📂 ফোল্ডার: {folder_name}", parse_mode=None)

    task_id = f"{folder_key}_{update.effective_chat.id}_{int(time.time())}"

    async with running_tasks_lock:
        running_tasks[task_id] = {
            "status": "running",
            "started_at": time.time(),
            "folder_key": folder_key,
            "folder_name": folder_name,
            "chat_id": update.effective_chat.id
        }

    async with task_controls_lock:
        task_controls[task_id] = {"cancel": False}

    asyncio.create_task(process_pdfs(folder_key, folder_name, update.effective_chat.id, task_id))

    await update.message.reply_text("✅ প্রসেসিং শুরু হয়েছে!\nসমাপ্ত হলে আমি জানিয়ে দেব।\n/cancel দিয়ে বন্ধ করতে পারেন।\n/status দিয়ে অগ্রগতি দেখুন।")

async def tg_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_key = context.user_data.get('folder_key')
    if not folder_key:
        await update.message.reply_text("❌ কোনো সক্রিয় প্রসেস নেই।")
        return

    task_id_to_cancel = None
    async with running_tasks_lock:
        for tid, task in running_tasks.items():
            if task.get("folder_key") == folder_key and task.get("status") == "running":
                task_id_to_cancel = tid
                break

    if task_id_to_cancel:
        async with task_controls_lock:
            if task_id_to_cancel in task_controls:
                task_controls[task_id_to_cancel]["cancel"] = True
                await update.message.reply_text("⛔ প্রসেস বন্ধ করার অনুরোধ পাঠানো হয়েছে...", parse_mode=None)
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

    checkpoint = await load_checkpoint(folder_key)
    await update.message.reply_text(
        f"📊 বর্তমান অবস্থা:\n✅ প্রসেস হয়েছে: {len(checkpoint.get('processed', []))}টি PDF\n"
        f"🔄 চলমান: {checkpoint.get('current', 'কিছু না')}\n"
        f"📄 শেষ পৃষ্ঠা: {checkpoint.get('last_page', 0)}",
        parse_mode=None
    )

# ============ Telegram Bot Setup ============
async def setup_bot():
    global application, telegram_worker_task, checkpoint_writer_task, disk_cleanup_task
    global _bot_initialized, _updater_started

    async with _bot_init_lock:
        if _bot_initialized:
            print("⚠️ বট আগেই চালু আছে, স্কিপ করা হচ্ছে")
            return

        print("🔧 setup_bot() শুরু হয়েছে", flush=True)

        telegram_worker_task = asyncio.create_task(telegram_worker())
        checkpoint_writer_task = asyncio.create_task(checkpoint_writer_worker())
        disk_cleanup_task = asyncio.create_task(disk_cleanup_worker())

        if not TELEGRAM_BOT_TOKEN:
            print("⚠️ TELEGRAM_BOT_TOKEN সেট করা নেই।")
            return

        # আগের ইনস্ট্যান্স থাকলে ক্লিনআপ
        if application is not None:
            try:
                await application.shutdown()
            except:
                pass
            application = None

        # নতুন অ্যাপ্লিকেশন তৈরি
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        application.add_handler(CommandHandler('start', tg_start))
        application.add_handler(CommandHandler('confirm', tg_confirm))
        application.add_handler(CommandHandler('cancel', tg_cancel))
        application.add_handler(CommandHandler('status', tg_status))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, tg_handle_link))

        # ইনিশিয়ালাইজ এবং স্টার্ট
        await application.initialize()
        await application.start()

        # শুধুমাত্র একবার polling শুরু করার ব্যবস্থা
        if not _updater_started:
            # পুরনো webhook রিমুভ করে polling শুরু
            try:
                await application.bot.delete_webhook(drop_pending_updates=True)
                await asyncio.sleep(1)  # টেলিগ্রামকে আপডেট হতে সময় দিন
            except Exception as e:
                print(f"Webhook ডিলিট করতে সমস্যা: {e}")

            # Polling শুরু (non-blocking)
            asyncio.create_task(application.updater.start_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True  # পুরনো আপডেট বাদ দিন
            ))
            _updater_started = True
            print("🤖 Telegram বট চলছে (ক্লিন অ্যাসিঙ্ক মোড)")
        else:
            print("⚠️ Polling আগেই চলছে, আবার শুরু করা হয়নি")

        _bot_initialized = True


async def shutdown_bot():
    global shutdown_in_progress, application, telegram_worker_task, checkpoint_writer_task, disk_cleanup_task
    global _bot_initialized, _updater_started

    shutdown_in_progress = True
    print("🛑 বন্ধ করা হচ্ছে...")

    # Updater বন্ধ করুন
    if application and hasattr(application, 'updater') and application.updater:
        try:
            await application.updater.stop()
        except Exception as e:
            print(f"Updater বন্ধ করতে সমস্যা: {e}")

    # টাস্কগুলো বাতিল করুন
    if telegram_worker_task:
        await telegram_queue.put(None)
        telegram_worker_task.cancel()
        try:
            await telegram_worker_task
        except asyncio.CancelledError:
            pass

    if checkpoint_writer_task:
        await checkpoint_write_queue.put(None)
        checkpoint_writer_task.cancel()
        try:
            await checkpoint_writer_task
        except asyncio.CancelledError:
            pass

    if disk_cleanup_task:
        disk_cleanup_task.cancel()
        try:
            await disk_cleanup_task
        except asyncio.CancelledError:
            pass

    if application:
        try:
            await application.stop()
            await application.shutdown()
            print("🤖 Telegram বট বন্ধ হয়েছে")
        except Exception as e:
            print(f"বট বন্ধ করতে সমস্যা: {e}")

    _bot_initialized = False
    _updater_started = False

# ============ FastAPI অ্যাপ ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_bot()
    yield
    await shutdown_bot()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Tafsir Image Processor is running"}

@app.post("/start_processing")
async def start_processing(request: ProcessRequest):
    if not request.folder_key or not request.folder_name:
        raise HTTPException(status_code=400, detail="folder_key and folder_name required")

    task_id = f"{request.folder_key}_{request.telegram_chat_id}_{int(time.time())}"

    async with running_tasks_lock:
        running_tasks[task_id] = {
            "status": "running",
            "started_at": time.time(),
            "folder_key": request.folder_key,
            "folder_name": request.folder_name,
            "chat_id": request.telegram_chat_id
        }

    async with task_controls_lock:
        task_controls[task_id] = {"cancel": False}

    asyncio.create_task(process_pdfs(request.folder_key, request.folder_name, request.telegram_chat_id, task_id))

    return {"status": "started", "message": "Processing started", "task_id": task_id}

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    async with task_controls_lock:
        if task_id in task_controls:
            task_controls[task_id]["cancel"] = True
            return {"status": "cancelling", "message": "Task cancellation requested"}
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/status/{folder_key}")
async def get_status(folder_key: str):
    checkpoint = await load_checkpoint(folder_key)
    return {
        "processed": len(checkpoint.get('processed', [])),
        "current": checkpoint.get('current'),
        "last_page": checkpoint.get('last_page', 0)
    }

@app.get("/tasks")
async def get_tasks():
    async with running_tasks_lock:
        return {k: v for k, v in running_tasks.items()}

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)