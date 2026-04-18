#!/usr/bin/env python3
"""
একক ফাইল: FastAPI + Telegram Bot (Threading Support)
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

# ============ Threading Helper ============
def run_async_in_thread(coro, *args):
    """একটি আলাদা থ্রেডে async ফাংশন রান করুন"""
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
    except RuntimeError:
        raise
    except Exception as e:
        print(f"[Disk Check] Warning: {e}", flush=True)
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
                wait_time = 5 * (2 ** retry)
                print(f"[Telegram] Flood control, waiting {wait_time}s", flush=True)
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(2)
        except Exception as e:
            await asyncio.sleep(3 * (retry + 1))
    return False

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
                print(f"[Telegram] Failed to send message", flush=True)
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
        await send_telegram_safe(chat_id, message)
    except Exception as e:
        print(f"[Telegram] Queue error: {e}", flush=True)

# ============ Internet Archive: আইটেম থেকে PDF লিংক বের করা ============
async def extract_pdfs_from_archive_item(item_url: str, max_pdfs: int = 30):
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
        print(f"[Archive] Error: {e}", flush=True)
    return pdf_urls

# ============ PDF ডাউনলোড ও প্রসেসিং ============
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

async def process_single_pdf(pdf, clean_folder_name, folder_key, chat_id, checkpoint, task_id):
    sub_folder = str(pdf['number'])
    full_hf_path = f"{clean_folder_name}/{sub_folder}"
    await send_telegram_safe(chat_id, f"📄 প্রসেসিং: {pdf['name']}\n📁 {full_hf_path}")
    
    start_page = 0
    if checkpoint.get('current') == pdf['name']:
        start_page = checkpoint.get('last_page', 0)
    
    if not pdf.get('download_link'):
        await send_telegram_safe(chat_id, f"❌ ডাউনলোড লিংক নেই: {pdf['name']}")
        return 0, 0, False
    
    await send_telegram_safe(chat_id, f"⬇️ ডাউনলোড হচ্ছে: {pdf['name']}")
    
    try:
        pdf_path = await download_pdf_stream(pdf['download_link'])
    except Exception as e:
        await send_telegram_safe(chat_id, f"❌ ডাউনলোড ব্যর্থ: {str(e)[:100]}")
        return 0, 0, False
    
    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        await send_telegram_safe(chat_id, f"🖼️ {total_pages} পৃষ্ঠা আপলোড শুরু...")
        
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
            except:
                pass
            
            if page_num % 50 == 0:
                checkpoint['current'] = pdf['name']
                checkpoint['last_page'] = page_num
                await async_save_checkpoint(folder_key, checkpoint)
                await send_telegram_safe(chat_id, f"📊 {page_num+1}/{total_pages} পৃষ্ঠা")
        
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

async def process_pdf_urls(pdf_urls: list, folder_name: str, chat_id: int, task_id: str):
    print(f"[DEBUG] ========== PROCESS STARTED ==========", flush=True)
    print(f"[DEBUG] Folder: {folder_name}", flush=True)
    print(f"[DEBUG] PDF Count: {len(pdf_urls)}", flush=True)
    
    try:
        clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
        await send_telegram_safe(chat_id, f"🚀 প্রসেসিং শুরু!\n📁 {clean_folder_name}\n📚 {len(pdf_urls)}টি PDF")
        
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
                await send_telegram_safe(chat_id, f"⏭️ স্কিপ: {pdf_name}")
                continue
            
            pdf = {'name': pdf_name, 'number': pdf_number, 'download_link': url}
            await send_telegram_safe(chat_id, f"📖 [{idx+1}/{len(pdf_urls)}] {pdf_name}")
            
            try:
                pages_processed, total_pages, cancelled = await process_single_pdf(
                    pdf, clean_folder_name, f"direct_{clean_folder_name}", chat_id, checkpoint, task_id
                )
                if cancelled:
                    break
                
                processed.append(pdf_name)
                checkpoint['processed'] = list(set(checkpoint.get('processed', []) + [pdf_name]))
                await async_save_checkpoint(f"direct_{clean_folder_name}", checkpoint)
                await send_telegram_safe(chat_id, f"✅ সম্পন্ন: {pdf_name} ({total_pages} পৃষ্ঠা)")
            except Exception as e:
                await send_telegram_safe(chat_id, f"❌ ব্যর্থ: {pdf_name}")
        
        if processed:
            await send_telegram_safe(chat_id, f"🎉 সব সম্পন্ন!\n📁 {HF_DATASET}/{clean_folder_name}\n✅ {len(processed)}টি PDF")
        
        async with running_tasks_lock:
            if task_id in running_tasks:
                running_tasks[task_id]["status"] = "completed"
    except Exception as e:
        print(f"[DEBUG] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        await send_telegram_safe(chat_id, f"❌ ত্রুটি: {str(e)[:200]}")

# ============ টেলিগ্রাম বট হ্যান্ডলার ============
async def tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 PDF to HuggingFace Processor\n\n"
        "প্রথম লাইনে বইয়ের নাম\n"
        "তারপর লিংক\n"
        "তারপর /confirm"
    )

async def tg_handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await update.message.reply_text("🔍 PDF লিংক বের করা হচ্ছে...")
        pdf_urls = await extract_pdfs_from_archive_item(archive_item)
        if pdf_urls:
            pdf_urls.sort(key=lambda x: int(re.search(r'/(\d+)\.pdf', x).group(1)) if re.search(r'/(\d+)\.pdf', x) else 999)
            await update.message.reply_text(f"✅ {len(pdf_urls)}টি PDF পাওয়া গেছে!")
    
    if pdf_urls:
        context.user_data['pdf_urls'] = pdf_urls
        context.user_data['folder_name'] = book_name or f"tafsir_{int(time.time())}"
        
        preview = "\n".join(pdf_urls[:5])
        if len(pdf_urls) > 5:
            preview += f"\n... এবং আরো {len(pdf_urls) - 5}টি"
        
        await update.message.reply_text(
            f"📚 বই: {context.user_data['folder_name']}\n"
            f"📄 {len(pdf_urls)}টি PDF\n\n{preview}\n\n✅ /confirm"
        )
        return
    
    await update.message.reply_text("❌ কোনো PDF লিংক পাওয়া যায়নি!")

async def tg_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[DEBUG] tg_confirm called", flush=True)
    
    pdf_urls = context.user_data.get('pdf_urls')
    folder_name = context.user_data.get('folder_name')
    
    if pdf_urls:
        if not folder_name:
            folder_name = f"tafsir_{int(time.time())}"
        
        await update.message.reply_text(f"🚀 শুরু হচ্ছে...\n📚 {len(pdf_urls)} PDF\n📁 {folder_name}")
        
        task_id = f"direct_{update.effective_chat.id}_{int(time.time())}"
        async with running_tasks_lock:
            running_tasks[task_id] = {"status": "running", "started_at": time.time()}
        async with task_controls_lock:
            task_controls[task_id] = {"cancel": False}
        
        # 🔥 আলাদা থ্রেডে প্রসেসিং শুরু করুন
        thread = threading.Thread(
            target=run_async_in_thread,
            args=(process_pdf_urls, pdf_urls, folder_name, update.effective_chat.id, task_id),
            daemon=True
        )
        thread.start()
        print(f"[DEBUG] Thread started for task {task_id}", flush=True)
        
        await update.message.reply_text("✅ শুরু হয়েছে! /status দেখুন")
        context.user_data['pdf_urls'] = None
        return
    
    await update.message.reply_text("❌ আগে PDF লিংক দিন")

async def tg_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    folder_name = context.user_data.get('folder_name', 'tafsir')
    clean_name = folder_name.replace(' ', '_')
    checkpoint = await load_checkpoint(f"direct_{clean_name}")
    
    processed = checkpoint.get('processed', [])
    current = checkpoint.get('current', 'কিছু না')
    last_page = checkpoint.get('last_page', 0)
    
    await update.message.reply_text(
        f"📊 অবস্থা:\n✅ প্রসেস: {len(processed)}টি PDF\n🔄 চলমান: {current}\n📄 শেষ পৃষ্ঠা: {last_page}"
    )

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
        application.add_handler(CommandHandler('confirm', tg_confirm))
        application.add_handler(CommandHandler('cancel', tg_cancel))
        application.add_handler(CommandHandler('status', tg_status))
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
    return {"status": "ok"}

@app.get("/tasks")
async def get_tasks():
    async with running_tasks_lock:
        return {k: v for k, v in running_tasks.items()}

@app.get("/start/{folder_name}")
async def start_processing_simple(folder_name: str):
    """সরাসরি URL দিয়ে প্রসেসিং শুরু করুন"""
    try:
        # আপনার PDF লিংকগুলো
        pdf_urls = [
            "https://archive.org/download/20260415_20260415_0945/1.pdf",
            "https://archive.org/download/20260415_20260415_0945/2.pdf",
            # ... বাকি 22 পর্যন্ত
        ]
        
        task_id = f"direct_api_{int(time.time())}"
        
        # থ্রেডে প্রসেসিং শুরু
        thread = threading.Thread(
            target=run_async_in_thread,
            args=(process_pdf_urls, pdf_urls, folder_name, 0, task_id),
            daemon=True
        )
        thread.start()
        
        return {
            "status": "started",
            "message": f"Processing {len(pdf_urls)} PDFs",
            "folder": folder_name,
            "task_id": task_id
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/start-one/{folder_name}")
async def start_one_pdf(folder_name: str):
    """শুধু ১টি PDF দিয়ে টেস্ট করুন"""
    try:
        pdf_urls = ["https://archive.org/download/20260415_20260415_0945/1.pdf"]
        
        task_id = f"test_{int(time.time())}"
        
        thread = threading.Thread(
            target=run_async_in_thread,
            args=(process_pdf_urls, pdf_urls, folder_name, 0, task_id),
            daemon=True
        )
        thread.start()
        
        return {
            "status": "started",
            "message": f"Processing 1 PDF",
            "folder": folder_name
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
