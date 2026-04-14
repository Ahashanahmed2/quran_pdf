import os
import asyncio
import io
import re
import json
import hashlib
import gc
import httpx
from datetime import datetime
from contextlib import asynccontextmanager
from urllib.parse import urlparse, parse_qs
from fastapi import FastAPI, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from pypdf import PdfReader
from pinecone import Pinecone, Index
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, login, hf_hub_download
from internetarchive import configure, upload
import logging

# ✅ OCR লাইব্রেরি
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available. Install: pytesseract, pdf2image, Pillow")

if OCR_AVAILABLE:
    try:
        pytesseract.get_tesseract_version()
    except:
        OCR_AVAILABLE = False
        print("⚠️ Tesseract not properly installed")

# --- ১. লগিং সেটআপ ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ২. Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- ৩. কনফিগারেশন ---
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quranqpf")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://quran-pdf-3.onrender.com")
SECRET_TOKEN = os.environ["WEBHOOK_SECRET"]
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IA_EMAIL = os.environ.get("IA_EMAIL", "")
IA_PASSWORD = os.environ.get("IA_PASSWORD", "")
GOOGLE_DRIVE_API_KEY = os.environ.get("GOOGLE_DRIVE_API_KEY", "")

HF_TRACKING_REPO = "ahashanahmed/quran-bot-tracking"

# ✅ OCR কনফিগারেশন
OCR_DPI = 300
OCR_LANG_PRIMARY = "ben+eng"
OCR_LANG_FALLBACK = "eng"

# ✅ Worker Queue System
pdf_processing_queue = None
MAX_WORKERS = 2
MAX_CONCURRENT_PER_USER = 3
QUEUE_MAX_SIZE = 20
SEARCH_THRESHOLD = 0.25  # ✅ Increased from 0.05

# ✅ গ্লোবাল ভেরিয়েবল
global_state = None
hf_cache = None
index: Index = None
embedding_model: SentenceTransformer = None
hf_api: HfApi = None
pc: Pinecone = None
startup_complete = False
startup_lock = asyncio.Lock()
index_creation_lock = asyncio.Lock()
telegram_bot: Bot = None
telegram_app: Application = None

# --- ৪. Thread-safe In-Memory State (⚠️ single-instance only) ---
class InMemoryState:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._processing_tasks = set()
        self._user_limits = {}
        self._active_workers = 0
        self._ip_requests = {}
    
    async def add_processing_task(self, task_id):
        async with self._lock:
            if task_id in self._processing_tasks:
                return False
            self._processing_tasks.add(task_id)
            return True
    
    async def remove_processing_task(self, task_id):
        async with self._lock:
            self._processing_tasks.discard(task_id)
    
    async def get_processing_count(self):
        async with self._lock:
            return len(self._processing_tasks)
    
    async def increment_user_tasks(self, user_id, max_allowed):
        async with self._lock:
            current = self._user_limits.get(user_id, 0)
            if current >= max_allowed:
                return False
            self._user_limits[user_id] = current + 1
            return True
    
    async def decrement_user_tasks(self, user_id):
        async with self._lock:
            if user_id in self._user_limits:
                self._user_limits[user_id] = max(0, self._user_limits[user_id] - 1)
    
    async def increment_workers(self):
        async with self._lock:
            self._active_workers += 1
            return self._active_workers
    
    async def decrement_workers(self):
        async with self._lock:
            self._active_workers = max(0, self._active_workers - 1)
            return self._active_workers
    
    async def get_worker_count(self):
        async with self._lock:
            return self._active_workers
    
    async def check_ip_rate_limit(self, ip, max_per_minute=30):
        async with self._lock:
            now = datetime.now()
            if ip not in self._ip_requests:
                self._ip_requests[ip] = []
            self._ip_requests[ip] = [t for t in self._ip_requests[ip] if (now - t).seconds < 60]
            if len(self._ip_requests[ip]) >= max_per_minute:
                return False
            self._ip_requests[ip].append(now)
            return True

class HFCache:
    def __init__(self):
        self._cache = None
        self._lock = asyncio.Lock()
    
    async def get(self):
        async with self._lock:
            return self._cache
    
    async def set(self, value):
        async with self._lock:
            self._cache = value

# --- ৫. Pinecone Index Creation (idempotent) ---
async def ensure_pinecone_index():
    global pc, index, PINECONE_INDEX_NAME, index_creation_lock
    
    async with index_creation_lock:
        try:
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing_indexes:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
                logger.info(f"✅ Created new Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                logger.info(f"✅ Pinecone index already exists: {PINECONE_INDEX_NAME}")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"✅ Pinecone index already exists: {PINECONE_INDEX_NAME}")
            else:
                raise

# --- ৬. Pinecone Upsert with Retry ---
async def pinecone_upsert_with_retry(vectors, max_retries=3):
    global index
    if index is None:
        raise RuntimeError("Pinecone index not ready")
    
    for attempt in range(max_retries):
        try:
            await asyncio.to_thread(index.upsert, vectors=vectors)
            return True
        except Exception as e:
            logger.warning(f"Pinecone upsert attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise

# --- ৭. SHA256 Hash for Duplicate Detection ---
def get_hybrid_hash(pdf_bytes):
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
    
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text() or ""
            text_hash = hashlib.sha256(first_page_text[:1000].encode()).hexdigest()
            return f"{content_hash[:8]}_{text_hash[:8]}"
    except:
        pass
    
    return content_hash[:16]

# --- ৮. Improved Heading Detection ---
def detect_headlines(page_text):
    headlines = []
    lines = page_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        score = 0
        
        if line.isupper():
            score += 4
        elif line.istitle() and len(line.split()) <= 5:
            score += 2
        if re.match(r'^[\d\.]+\s+\w+', line):
            score += 4
        if len(line) < 60:
            score += 2
        if re.search(r'(chapter|section|part|lesson|unit|module|অধ্যায়|পরিচ্ছেদ)', line, re.IGNORECASE):
            score += 3
        if re.match(r'^[=\-]{2,}.*[=\-]{2,}$', line):
            score += 5
        
        if score >= 6:
            headlines.append(line)
    
    return headlines

# --- ৯. HF ট্র্যাকিং ফাংশন ---
async def load_upload_history():
    global hf_cache
    cache = await hf_cache.get()
    if cache is not None:
        return cache
    
    try:
        if hf_api:
            file_path = await asyncio.wait_for(
                asyncio.to_thread(
                    hf_hub_download,
                    repo_id=HF_TRACKING_REPO,
                    filename="uploaded_files.json",
                    repo_type="dataset",
                    token=HF_TOKEN
                ),
                timeout=30
            )
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                await hf_cache.set(data)
                return data
    except Exception as e:
        logger.warning(f"HF থেকে লোড করা যায়নি: {e}")
    
    default_data = {"files": {}, "total_uploads": 0}
    await hf_cache.set(default_data)
    return default_data

async def save_upload_history(history):
    global hf_cache
    await hf_cache.set(history)
    
    for attempt in range(3):
        try:
            if hf_api:
                temp_path = "/tmp/uploaded_files.json"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                
                await asyncio.wait_for(
                    asyncio.to_thread(
                        hf_api.upload_file,
                        path_or_fileobj=temp_path,
                        path_in_repo="uploaded_files.json",
                        repo_id=HF_TRACKING_REPO,
                        repo_type="dataset",
                        commit_message=f"Update tracking - {datetime.now().isoformat()}"
                    ),
                    timeout=60
                )
                logger.info("✅ Tracking uploaded to Hugging Face")
                return True
        except Exception as e:
            logger.warning(f"HF upload attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    
    logger.error("HF upload failed after 3 attempts")
    return False

async def check_file_already_uploaded(file_hash):
    history = await load_upload_history()
    if file_hash in history.get('files', {}):
        return {
            'exists': True,
            'date': history['files'][file_hash].get('uploaded_at', 'Unknown'),
            'archive_url': history['files'][file_hash].get('archive_url', ''),
            'filename': history['files'][file_hash].get('filename', '')
        }
    return {'exists': False}

async def mark_file_as_uploaded(filename, file_hash, archive_url, pages, vectors, size_mb, volume_info=None):
    history = await load_upload_history()
    history['files'][file_hash] = {
        'filename': filename,
        'hash': file_hash,
        'uploaded_at': datetime.now().isoformat(),
        'archive_url': archive_url,
        'pages': pages,
        'vectors': vectors,
        'size_mb': size_mb,
        'volume_info': volume_info or {}
    }
    history['total_uploads'] = len(history['files'])
    history['last_updated'] = datetime.now().isoformat()
    await save_upload_history(history)

# --- ১০. Archive.org ফাংশন ---
def upload_to_archive_sync(pdf_bytes, filename, title=""):
    try:
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:10]
        
        safe_filename = re.sub(r'[^\x20-\x7E]', '_', filename)
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', safe_filename)
        safe_filename = re.sub(r'_+', '_', safe_filename)
        safe_filename = safe_filename.strip('_')
        safe_filename = safe_filename.replace('\x00', '')
        
        if not safe_filename or safe_filename == '.pdf':
            safe_filename = f"document_{file_hash}.pdf"
        elif not safe_filename.endswith('.pdf'):
            safe_filename += '.pdf'
        
        identifier = f"quran_bot_{file_hash}_{safe_filename.replace('.pdf', '')}"
        
        metadata = {
            'title': title or filename,
            'mediatype': 'texts',
            'collection': 'opensource',
            'description': f"Uploaded via Telegram Quran Bot on {datetime.now().strftime('%Y-%m-%d')}",
            'subject': ['Quran', 'PDF', 'Telegram Bot'],
            'language': 'ben'
        }
        
        response = upload(
            identifier=identifier,
            files={safe_filename: pdf_bytes},
            metadata=metadata,
        )
        
        if hasattr(response[0], 'status_code') and response[0].status_code == 200:
            logger.info(f"✅ Archive.org upload: {identifier}")
            return {
                'success': True,
                'identifier': identifier,
                'url': f"https://archive.org/details/{identifier}",
                'pdf_url': f"https://archive.org/download/{identifier}/{safe_filename}"
            }
        else:
            logger.error(f"Archive.org upload failed")
            return {'success': False, 'error': 'Upload failed'}
            
    except Exception as e:
        logger.error(f"Archive.org error: {e}")
        return {'success': False, 'error': str(e)}

async def upload_to_archive(pdf_bytes, filename, title=""):
    return await asyncio.to_thread(upload_to_archive_sync, pdf_bytes, filename, title)

# --- ১১. Google Drive ফাংশন ---
def extract_folder_id_from_url(url):
    match = re.search(r'/drive/folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1), 'folder'
    
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1), 'file'
    
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if 'id' in params:
        return params['id'][0], 'file'
    
    return None, None

def parse_volume_info(text):
    volume_info = {}
    
    book_match = re.search(r'book[=:\s]+([^\n]+?)(?:\s+volume|\s*$)', text, re.IGNORECASE)
    if book_match:
        volume_info['book_name'] = book_match.group(1).strip()
    
    volume_match = re.search(r'volume[=:\s]+(\d+)', text, re.IGNORECASE)
    if volume_match:
        volume_info['volume'] = int(volume_match.group(1))
    
    return volume_info

async def download_with_retry(client, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with client.stream("GET", url, timeout=120.0) as response:
                if response.status_code == 200:
                    content = await response.aread()
                    return type("Response", (), {"status_code": 200, "content": content})()
                logger.warning(f"Download attempt {attempt + 1} returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
        
        if attempt == max_retries - 1:
            raise Exception(f"Download failed after {max_retries} attempts")
        
        await asyncio.sleep(2 ** attempt)
    
    raise Exception("Download failed")

# --- ১২. PDF প্রসেসিং ফাংশন ---
def detect_para_nesting(para_text):
    nesting_level = 0
    if para_text.startswith((' ', '\t')):
        leading_spaces = len(para_text) - len(para_text.lstrip())
        nesting_level = leading_spaces // 4
    if re.match(r'^[\d]+[\.\)]', para_text):
        nesting_level += 1
    elif para_text.startswith(('•', '-', '*', '○', '▪')):
        nesting_level += 1
    return min(nesting_level, 3)

def extract_paragraphs_with_full_hierarchy(page_text, page_num, chapter, book, volume, 
                                          global_counter, prev_continued, prev_text, 
                                          prev_page, prev_para_num):
    raw_paras = re.split(r'\n\s*\n', page_text)
    paragraphs = []
    local_para_counter = 1
    
    for idx, para in enumerate(raw_paras):
        para = para.strip()
        if len(para) < 25:
            continue
        
        para = re.sub(r'\s+', ' ', para)
        
        para_id = f"{book}_{volume}_{chapter}_{page_num}_{local_para_counter}"
        para_id = re.sub(r'[^a-zA-Z0-9_]', '_', para_id)[:100]
        
        nesting_level = detect_para_nesting(para)
        parent_para_id = None
        
        if nesting_level > 0 and paragraphs:
            for prev_para in reversed(paragraphs):
                if prev_para['nesting_level'] < nesting_level:
                    parent_para_id = prev_para['para_id']
                    break
        
        continues_to_next = False
        if idx == len(raw_paras) - 1:
            if not para.endswith(('.', '!', '?', '।', '”', '"')):
                continues_to_next = True
        
        continued_from_prev = prev_continued and idx == 0
        
        full_para_text = para
        if continued_from_prev and prev_text:
            full_para_text = prev_text + " " + para
        
        paragraphs.append({
            'para_id': para_id,
            'text': para,
            'full_continued_text': full_para_text,
            'page': page_num,
            'para_number': local_para_counter,
            'global_para_number': global_counter + local_para_counter,
            'chapter': chapter,
            'book': book,
            'volume': volume,
            'nesting_level': nesting_level,
            'parent_para_id': parent_para_id,
            'continues_to_next': continues_to_next,
            'continued_from_prev': continued_from_prev,
            'prev_para_page': prev_page if continued_from_prev else None,
            'prev_para_number': prev_para_num if continued_from_prev else None,
            'total_paras_on_page': 0,
            'char_count': len(para),
            'word_count': len(para.split())
        })
        
        local_para_counter += 1
    
    for para in paragraphs:
        para['total_paras_on_page'] = len(paragraphs)
    
    return paragraphs

def finalize_paragraph_hierarchy(structured_pages):
    para_hierarchy = {}
    
    for page in structured_pages:
        for para in page['paragraphs']:
            if para.get('parent_para_id'):
                para_hierarchy[para['para_id']] = para['parent_para_id']
    
    for page in structured_pages:
        for para in page['paragraphs']:
            para['has_parent'] = para.get('parent_para_id') is not None
            
            child_paras = []
            for child_id, parent_id in para_hierarchy.items():
                if parent_id == para['para_id']:
                    child_paras.append(child_id)
            
            if child_paras:
                para['child_para_ids'] = child_paras
                para['has_children'] = True
            else:
                para['has_children'] = False
    
    return structured_pages

def ocr_page_sync(image):
    """Synchronous OCR for a single page"""
    try:
        return pytesseract.image_to_string(image, lang=OCR_LANG_PRIMARY)
    except:
        try:
            return pytesseract.image_to_string(image, lang=OCR_LANG_FALLBACK)
        except:
            return ""

async def process_pdf_with_progress(pdf_bytes, filename, volume_info, status_msg, bot, chat_id):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    
    book_name = volume_info.get('book_name', filename.replace('.pdf', ''))
    volume_number = volume_info.get('volume', 1)
    
    first_page_text = ""
    if total_pages > 0:
        first_page_text = await asyncio.to_thread(reader.pages[0].extract_text) or ""
    needs_ocr = len(first_page_text.strip()) < 50
    
    structured_pages = []
    current_chapter = "মূল অংশ"
    prev_para_continued = False
    prev_para_text = ""
    prev_para_page = 0
    prev_para_number = 0
    global_para_counter = 0
    
    batch_size = 10
    
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        
        progress_text = f"📊 প্রক্রিয়াকরণ: ব্যাচ {batch_start//batch_size + 1}/{(total_pages + batch_size - 1)//batch_size}\n"
        progress_text += f"📄 পৃষ্ঠা: {batch_start + 1}-{batch_end} / {total_pages}"
        
        try:
            await status_msg.edit_text(progress_text)
        except Exception as e:
            logger.warning(f"Failed to edit progress: {e}")
        
        for page_num in range(batch_start + 1, batch_end + 1):
            if needs_ocr and OCR_AVAILABLE:
                images = await asyncio.to_thread(
                    convert_from_bytes, pdf_bytes, 
                    dpi=OCR_DPI, first_page=page_num, last_page=page_num
                )
                if images:
                    page_text = ""
                    for img in images:
                        try:
                            page_text = await asyncio.to_thread(ocr_page_sync, img)
                        finally:
                            # ✅ Proper PIL cleanup
                            if hasattr(img, 'fp') and img.fp:
                                img.fp.close()
                            img.close()
                    images.clear()
                    del images
                else:
                    page_text = ""
            else:
                page = reader.pages[page_num - 1]
                page_text = await asyncio.to_thread(page.extract_text) or ""
            
            if page_text and page_text.strip():
                headlines = detect_headlines(page_text)
                if headlines:
                    current_chapter = headlines[0][:100]
                
                paragraphs = extract_paragraphs_with_full_hierarchy(
                    page_text, page_num, current_chapter, book_name, volume_number,
                    global_para_counter, prev_para_continued, prev_para_text,
                    prev_para_page, prev_para_number
                )
                
                if paragraphs:
                    last_para = paragraphs[-1]
                    prev_para_continued = last_para.get('continues_to_next', False)
                    if prev_para_continued:
                        prev_para_text = last_para['text']
                        prev_para_page = page_num
                        prev_para_number = last_para['para_number']
                    global_para_counter += len(paragraphs)
                
                structured_pages.append({
                    'page_number': page_num,
                    'headlines': headlines,
                    'paragraphs': paragraphs,
                    'chapter': current_chapter,
                    'book': book_name,
                    'volume': volume_number,
                    'full_text': page_text,
                    'para_count': len(paragraphs),
                    'has_continuation': prev_para_continued
                })
            else:
                structured_pages.append({
                    'page_number': page_num,
                    'headlines': [],
                    'paragraphs': [],
                    'chapter': current_chapter,
                    'book': book_name,
                    'volume': volume_number,
                    'full_text': "",
                    'para_count': 0,
                    'has_continuation': prev_para_continued
                })
        
        if batch_start % (batch_size * 5) == 0:
            gc.collect()
    
    return finalize_paragraph_hierarchy(structured_pages)

def create_structured_chunks_hierarchical(structured_pages, filename):
    chunks = []
    
    for page in structured_pages:
        page_num = page['page_number']
        headlines = page['headlines']
        paragraphs = page['paragraphs']
        chapter = page.get('chapter', 'Unknown')
        book = page.get('book', filename)
        volume = page.get('volume', 1)
        
        for headline in headlines:
            chunks.append({
                'text': headline,
                'metadata': {
                    'type': 'headline',
                    'page': page_num,
                    'headline': headline[:100],
                    'chapter': chapter,
                    'book': book,
                    'volume': volume
                }
            })
        
        for para in paragraphs:
            metadata = {
                'type': 'paragraph',
                'para_id': para.get('para_id', ''),
                'page': page_num,
                'para_number': para.get('para_number', 1),
                'global_para_number': para.get('global_para_number', 1),
                'headline': headlines[0][:100] if headlines else "No headline",
                'chapter': chapter,
                'book': book,
                'volume': volume,
                'total_paras_on_page': para.get('total_paras_on_page', 0),
                'nesting_level': para.get('nesting_level', 0),
                'parent_para_id': para.get('parent_para_id'),
                'has_parent': para.get('has_parent', False),
                'has_children': para.get('has_children', False),
                'continues_to_next': para.get('continues_to_next', False),
                'continued_from_prev': para.get('continued_from_prev', False),
                'char_count': para.get('char_count', 0),
                'word_count': para.get('word_count', 0)
            }
            
            if para.get('child_para_ids'):
                metadata['child_para_ids'] = para['child_para_ids']
            
            chunks.append({
                'text': para.get('full_continued_text', para['text']),
                'metadata': metadata
            })
        
        if page.get('full_text', '').strip():
            chunks.append({
                'text': page['full_text'][:2000],
                'metadata': {
                    'type': 'full_page',
                    'page': page_num,
                    'headline': headlines[0][:100] if headlines else "No headline",
                    'chapter': chapter,
                    'book': book,
                    'volume': volume,
                    'para_count': page.get('para_count', 0)
                }
            })
    
    return chunks

async def save_structured_to_pinecone(filename, chunks):
    global index, embedding_model
    if index is None or embedding_model is None:
        raise RuntimeError("Pinecone not ready")
    
    batch_size = 25
    file_hash = hashlib.sha256(filename.encode()).hexdigest()[:8]
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        texts = [chunk['text'] for chunk in batch_chunks]
        
        embeddings = await asyncio.to_thread(
            embedding_model.encode, texts, 
            batch_size=64, show_progress_bar=False
        )
        embeddings = embeddings.tolist()
        
        vectors = []
        for j, (chunk_data, embedding) in enumerate(zip(batch_chunks, embeddings)):
            metadata = chunk_data['metadata']
            
            full_metadata = {
                "filename": filename,
                "chunk_index": i + j,
                "text": chunk_data['text'][:1000],
                **metadata
            }
            
            vectors.append({
                "id": f"{file_hash}_{i + j}",
                "values": embedding,
                "metadata": full_metadata
            })
        
        await pinecone_upsert_with_retry(vectors)
        logger.info(f"   📤 Uploaded batch {i//batch_size + 1} ({len(vectors)} vectors)")
    
    return len(chunks)

async def search_in_pinecone_advanced(query, top_k=5):
    global index, embedding_model
    if index is None or embedding_model is None:
        return []
    
    try:
        query_embedding = await asyncio.to_thread(embedding_model.encode, query)
        query_embedding = query_embedding.tolist()
        
        results = await asyncio.to_thread(
            index.query,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in results.matches:
            if match.score > SEARCH_THRESHOLD:
                metadata = match.metadata
                chunks.append({
                    "text": metadata.get("text", ""),
                    "filename": metadata.get("filename", ""),
                    "page": metadata.get("page", "N/A"),
                    "headline": metadata.get("headline", "N/A"),
                    "chapter": metadata.get("chapter", "N/A"),
                    "book": metadata.get("book", "N/A"),
                    "volume": metadata.get("volume", "N/A"),
                    "para_id": metadata.get("para_id", ""),
                    "para_number": metadata.get("para_number", "N/A"),
                    "nesting_level": metadata.get("nesting_level", 0),
                    "continues_to_next": metadata.get("continues_to_next", False),
                    "continued_from_prev": metadata.get("continued_from_prev", False),
                    "type": metadata.get("type", "text"),
                    "score": match.score
                })
        return chunks
    except Exception as e:
        logger.error(f"Pinecone সার্চ ত্রুটি: {e}")
        return []

def format_search_results(results, query):
    if not results:
        return "❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"
    
    formatted = f"🔍 প্রশ্ন: {query}\n\n📊 প্রাপ্ত ফলাফল: {len(results)}টি\n\n"
    
    books = {}
    for r in results:
        book_key = f"{r.get('book', 'Unknown')} (খণ্ড {r.get('volume', 'N/A')})"
        if book_key not in books:
            books[book_key] = []
        books[book_key].append(r)
    
    for book_key, book_results in books.items():
        formatted += f"📚 {book_key}:\n"
        
        for r in book_results[:3]:
            formatted += f"\n📍 পৃষ্ঠা {r['page']}"
            if r.get('chapter') and r['chapter'] != 'N/A':
                formatted += f" | {r['chapter'][:40]}"
            if r.get('para_number') and r['para_number'] != 'N/A':
                formatted += f" | প্যারা {r['para_number']}"
            if r.get('nesting_level', 0) > 0:
                formatted += f" | সাব-প্যারা (লেভেল {r['nesting_level']})"
            if r.get('continued_from_prev'):
                formatted += " | পূর্ববর্তী পৃষ্ঠা থেকে আগত"
            if r.get('continues_to_next'):
                formatted += " | পরবর্তী পৃষ্ঠায় চলমান"
            
            formatted += f"\n{r['text'][:300]}...\n"
    
    formatted += f"\n\n---\n📚 সোর্স: {results[0].get('filename', 'Unknown')}"
    
    if len(formatted) > 4000:
        formatted = formatted[:4000] + "..."
    
    return formatted

# --- ১৩. Worker System ---
async def pdf_worker():
    global pdf_processing_queue, global_state
    while True:
        try:
            task = await pdf_processing_queue.get()
            if task is None:
                break
            
            folder_id, chat_id, bot, volume_info = task
            
            await global_state.increment_workers()
            
            try:
                await process_single_pdf(folder_id, chat_id, bot, volume_info)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                try:
                    await bot.send_message(chat_id, f"❌ প্রক্রিয়াকরণে ত্রুটি: {str(e)}")
                except:
                    pass
            finally:
                await global_state.remove_processing_task(folder_id)
                await global_state.decrement_user_tasks(chat_id)
                await global_state.decrement_workers()
                pdf_processing_queue.task_done()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Worker loop error: {e}")

async def process_single_pdf(folder_id, chat_id, bot, volume_info):
    download_url = f"https://drive.google.com/uc?export=download&id={folder_id}"
    
    timeout_config = httpx.Timeout(120.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        response = await download_with_retry(client, download_url)
    
    if response.status_code != 200:
        await bot.send_message(chat_id, f"❌ ডাউনলোড ব্যর্থ: {response.status_code}")
        return
    
    pdf_bytes = response.content
    file_hash = get_hybrid_hash(pdf_bytes)
    
    duplicate = await check_file_already_uploaded(file_hash)
    if duplicate['exists']:
        await bot.send_message(
            chat_id,
            f"⚠️ এই ফাইলটি আগেই আপলোড করা হয়েছে!\n\n"
            f"📅 আগের আপলোড: {duplicate['date']}\n"
            f"🔗 Archive.org: {duplicate['archive_url']}"
        )
        return
    
    filename = f"drive_{file_hash[:8]}.pdf"
    size_mb = len(pdf_bytes) / 1024 / 1024
    
    archive_result = await upload_to_archive(pdf_bytes, filename)
    archive_url = archive_result.get('url') if archive_result['success'] else None
    
    status_msg = await bot.send_message(chat_id, "⏳ PDF প্রক্রিয়াকরণ শুরু...")
    
    structured_pages = await process_pdf_with_progress(
        pdf_bytes, filename, volume_info, status_msg, bot, chat_id
    )
    
    chunks = create_structured_chunks_hierarchical(structured_pages, filename)
    vector_count = await save_structured_to_pinecone(filename, chunks)
    
    total_pages = len(structured_pages)
    
    volume_summary = {
        'book_name': volume_info.get('book_name', filename),
        'volume': volume_info.get('volume', 1),
        'total_pages': total_pages
    }
    
    await mark_file_as_uploaded(filename, file_hash, archive_url, total_pages, vector_count, size_mb, volume_summary)
    
    final_msg = f"✅ আপনার PDF সফলভাবে সংরক্ষিত হয়েছে!\n\n"
    final_msg += f"📚 বই: {volume_summary['book_name']}\n"
    final_msg += f"📖 খণ্ড: {volume_summary['volume']}\n"
    final_msg += f"📄 পৃষ্ঠা: {total_pages}\n"
    final_msg += f"🗄️ ভেক্টর: {vector_count}\n"
    final_msg += f"📊 আকার: {size_mb:.2f} MB\n"
    
    if archive_url:
        final_msg += f"\n📚 Archive.org: {archive_url}\n\n"
    
    final_msg += "🎉 এখন আপনি এই PDF সম্পর্কে প্রশ্ন করতে পারেন!"
    
    await status_msg.edit_text(final_msg)

# --- ১৪. Telegram বট হ্যান্ডলার ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Quran PDF Bot\n\n"
        "/help - সকল কমান্ড দেখুন\n"
        "/status - সিস্টেম স্ট্যাটাস\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n\n"
        "✨ Google Drive লিংক দিন বড় PDF আপলোড করতে!\n\n"
        "📝 বই ও খণ্ড তথ্য দিতে:\n"
        "লিংকের সাথে লিখুন: book=বইয়ের নাম volume=1"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📚 উপলব্ধ কমান্ডসমূহ:\n\n"
        "/start - বট চালু করুন\n"
        "/help - এই সাহায্য বার্তা\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n"
        "/status - সিস্টেম স্ট্যাটাস\n\n"
        "Google Drive থেকে PDF আপলোড:\n"
        "1. PDF Google Drive-এ আপলোড করুন\n"
        "2. শেয়ারেবল লিংক কপি করুন\n"
        "3. লিংকটি এখানে পেস্ট করুন\n\n"
        "বই ও খণ্ড তথ্য দিতে (ঐচ্ছিক):\n"
        "লিংকের সাথে লিখুন: book=বইয়ের নাম volume=1\n\n"
        "প্রশ্ন: সরাসরি প্রশ্ন লিখুন"
    )
    await update.message.reply_text(help_text)

async def handle_drive_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global pdf_processing_queue, global_state, startup_complete
    
    if not startup_complete:
        await update.message.reply_text("⚠️ সিস্টেম প্রস্তুত হচ্ছে, অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।")
        return
    
    if global_state is None:
        await update.message.reply_text("⚠️ সিস্টেম প্রস্তুত হচ্ছে, অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।")
        return
    
    url = update.message.text
    logger.info(f"🚀 handle_drive_link started with URL: {url}")
    
    folder_id, url_type = extract_folder_id_from_url(url)
    
    if not folder_id:
        await update.message.reply_text("❌ অবৈধ Google Drive লিংক।")
        return
    
    if url_type == 'folder':
        await update.message.reply_text("📁 ফোল্ডার প্রক্রিয়াকরণ শীঘ্রই আসছে...")
        return
    
    chat_id = update.effective_chat.id
    
    # ✅ Queue full early reject
    if pdf_processing_queue.full():
        await update.message.reply_text("⚠️ সার্ভার ব্যস্ত, অনুগ্রহ করে ৫ মিনিট পর আবার চেষ্টা করুন।")
        return
    
    if not await global_state.add_processing_task(folder_id):
        await update.message.reply_text("⚠️ এই PDFটি ইতিমধ্যে প্রক্রিয়াকরণ হচ্ছে...")
        return
    
    if not await global_state.increment_user_tasks(chat_id, MAX_CONCURRENT_PER_USER):
        await global_state.remove_processing_task(folder_id)
        await update.message.reply_text(f"⚠️ আপনি ইতিমধ্যে {MAX_CONCURRENT_PER_USER}টি PDF প্রক্রিয়াকরণ করছেন।")
        return
    
    volume_info = parse_volume_info(url)
    
    await update.message.reply_text(
        "📥 আপনার PDF প্রক্রিয়াকরণের জন্য কিউতে যোগ হয়েছে!\n\n"
        f"📚 বই: {volume_info.get('book_name', 'Unknown')}\n"
        f"📖 খণ্ড: {volume_info.get('volume', 1)}\n\n"
        "⏳ স্ক্যান করা PDF হলে ২০-৩০ মিনিট সময় লাগতে পারে।\n"
        "✅ প্রক্রিয়া শেষে আপনাকে জানানো হবে।"
    )
    
    # ✅ Simple put without timeout
    await pdf_processing_queue.put((folder_id, chat_id, context.bot, volume_info))

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        results = await search_in_pinecone_advanced(user_question, top_k=5)
        formatted_answer = format_search_results(results, user_question)
        await update.message.reply_text(formatted_answer)
    except Exception as e:
        logger.error(f"Question handling error: {e}")
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def list_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        history = await load_upload_history()
        
        if history.get('files'):
            file_list = ""
            for file_hash, file_info in history['files'].items():
                file_list += f"\n📁 {file_info.get('filename', 'Unknown')}\n"
                volume_info = file_info.get('volume_info', {})
                file_list += f"   📚 {volume_info.get('book_name', 'Unknown')} (খণ্ড {volume_info.get('volume', 'N/A')})\n"
                file_list += f"   📄 পৃষ্ঠা: {file_info.get('pages', 'N/A')}\n"
                file_list += f"   🗄️ ভেক্টর: {file_info.get('vectors', 'N/A')}\n"
            
            await update.message.reply_text(f"সংরক্ষিত PDF:\n{file_list}")
        else:
            await update.message.reply_text("ℹ️ এখনো কোনো PDF সংরক্ষিত হয়নি।")
    except Exception as e:
        logger.error(f"List files error: {e}")
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global index, hf_api, pdf_processing_queue, global_state, startup_complete
    
    pc_status = "✅" if index is not None else "❌"
    hf_status = "✅" if hf_api is not None else "❌"
    ia_status = "✅" if IA_EMAIL and IA_PASSWORD else "❌"
    ocr_status = "✅" if OCR_AVAILABLE else "❌"
    startup = "✅" if startup_complete else "⏳"
    
    current_active = await global_state.get_worker_count() if global_state else 0
    current_processing = await global_state.get_processing_count() if global_state else 0
    
    status_text = (
        f"📊 সিস্টেম স্ট্যাটাস\n\n"
        f"🚀 Startup: {startup}\n"
        f"🗄️ Pinecone: {pc_status}\n"
        f"📚 Archive.org: {ia_status}\n"
        f"🔍 HF Tracking: {hf_status}\n"
        f"📷 OCR (DPI {OCR_DPI}): {ocr_status}\n"
        f"🌐 OCR ভাষা: {OCR_LANG_PRIMARY}\n"
        f"🔎 সার্চ থ্রেশহোল্ড: {SEARCH_THRESHOLD}\n"
        f"📁 ইনডেক্স: {PINECONE_INDEX_NAME}\n"
        f"👷 Active Workers: {current_active}/{MAX_WORKERS}\n"
        f"📋 Queue Size: {pdf_processing_queue.qsize() if pdf_processing_queue else 0}\n"
        f"🔄 Processing: {current_processing} tasks"
    )
    await update.message.reply_text(status_text)

# --- 15. FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pdf_processing_queue, global_state, hf_cache, startup_complete
    global index, embedding_model, hf_api, pc, telegram_bot, telegram_app
    
    logger.info("🚀 Starting initialization...")
    
    global_state = InMemoryState()
    hf_cache = HFCache()
    pdf_processing_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
            hf_api = HfApi(token=HF_TOKEN)
            hf_api.create_repo(repo_id=HF_TRACKING_REPO, repo_type="dataset", exist_ok=True)
            logger.info(f"✅ HF Tracking Repo ready: {HF_TRACKING_REPO}")
        except Exception as e:
            logger.error(f"❌ HF Tracking setup failed: {e}")
    
    if IA_EMAIL and IA_PASSWORD:
        try:
            configure(IA_EMAIL, IA_PASSWORD)
            logger.info("✅ Archive.org configured")
        except Exception as e:
            logger.error(f"❌ Archive.org setup failed: {e}")
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        await ensure_pinecone_index()
        index = pc.Index(PINECONE_INDEX_NAME)
        embedding_model = await asyncio.to_thread(SentenceTransformer, 'all-MiniLM-L6-v2')
        logger.info("✅ Pinecone ও এম্বেডিং মডেল লোড হয়েছে")
    except Exception as e:
        logger.error(f"❌ Pinecone সেটআপ ত্রুটি: {e}")
        index = None
        embedding_model = None
    
    workers = []
    for _ in range(MAX_WORKERS):
        worker = asyncio.create_task(pdf_worker())
        worker.add_done_callback(lambda t: logger.error(f"Worker crashed: {t.exception()}") if t.exception() else None)
        workers.append(worker)
    
    request = HTTPXRequest(connection_pool_size=10, read_timeout=120, write_timeout=120)
    telegram_bot = Bot(token=TELEGRAM_TOKEN, request=request)
    
    telegram_app = Application.builder().bot(telegram_bot).build()
    
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("list", list_files))
    telegram_app.add_handler(CommandHandler("status", status))
    telegram_app.add_handler(MessageHandler(filters.Regex(r'drive\.google\.com'), handle_drive_link))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_question))
    
    await telegram_app.initialize()
    app.state.ptb_app = telegram_app
    app.state.bot = telegram_bot
    
    webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
    await telegram_bot.set_webhook(url=webhook_url, secret_token=SECRET_TOKEN)
    logger.info(f"✅ Webhook set to: {webhook_url}")
    
    startup_complete = True
    logger.info("🎉 Initialization complete!")
    
    yield
    
    startup_complete = False
    
    for _ in range(len(workers)):
        try:
            await asyncio.wait_for(pdf_processing_queue.put(None), timeout=1)
        except:
            pass
    
    for worker in workers:
        worker.cancel()
    
    await asyncio.gather(*workers, return_exceptions=True)
    await telegram_bot.delete_webhook()
    await telegram_app.shutdown()

# --- ১৬. FastAPI App ---
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

hostname = RENDER_EXTERNAL_URL.replace("https://", "").replace("http://", "")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[hostname])

# --- ১৭. রুট এন্ডপয়েন্ট ---
@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    return {"status": "ok", "service": "Quran PDF Bot"}

@app.get("/healthcheck")
async def health():
    return {"status": "ok"}

@app.get("/ping")
@limiter.limit("60/minute")
async def ping(request: Request):
    return {"pong": True}

@app.post("/telegram-webhook")
@limiter.limit("60/minute")
async def telegram_webhook(request: Request):
    global global_state, startup_complete, telegram_bot, telegram_app
    
    if not startup_complete:
        return Response(status_code=503, content="Service starting")
    
    client_ip = get_remote_address(request)
    if global_state and not await global_state.check_ip_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return Response(status_code=429)
    
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != SECRET_TOKEN:
        logger.warning("❌ Invalid secret token")
        return Response(status_code=403)
    
    # ✅ Extract data first, then background
    try:
        data = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse webhook JSON: {e}")
        return Response(status_code=400)
    
    asyncio.create_task(process_webhook_update(data))
    return Response(status_code=200)

async def process_webhook_update(data):
    global telegram_bot, telegram_app
    try:
        update = Update.de_json(data, telegram_bot)
        await telegram_app.process_update(update)
    except Exception as e:
        logger.error(f"Webhook background processing error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)