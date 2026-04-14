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
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from pypdf import PdfReader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, login, hf_hub_download
from internetarchive import configure, upload
import logging

# ✅ OCR লাইব্রেরি
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available. Install: pytesseract, pdf2image")

# --- ১. লগিং সেটআপ ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ২. কনফিগারেশন (সব Environment Variable থেকে) ---
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

# ✅ ডুপ্লিকেট প্রসেসিং ও রেট লিমিট
processing_tasks = set()
user_limits = {}  # chat_id -> active_tasks_count
MAX_CONCURRENT_PER_USER = 3

# ✅ HF Cache
UPLOAD_CACHE = None

# --- ৩. Hugging Face ট্র্যাকিং সেটআপ ---
hf_api = None
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        hf_api = HfApi(token=HF_TOKEN)
        hf_api.create_repo(repo_id=HF_TRACKING_REPO, repo_type="dataset", exist_ok=True)
        logger.info(f"✅ HF Tracking Repo ready: {HF_TRACKING_REPO}")
    except Exception as e:
        logger.error(f"❌ HF Tracking setup failed: {e}")

# --- ৪. Archive.org সেটআপ ---
if IA_EMAIL and IA_PASSWORD:
    try:
        configure(IA_EMAIL, IA_PASSWORD)
        logger.info("✅ Archive.org configured")
    except Exception as e:
        logger.error(f"❌ Archive.org setup failed: {e}")

# --- ৫. Pinecone ও এম্বেডিং মডেল সেটআপ ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Pinecone ও এম্বেডিং মডেল লোড হয়েছে")
except Exception as e:
    logger.error(f"❌ Pinecone সেটআপ ত্রুটি: {e}")
    index = None
    embedding_model = None

# --- ৬. HF ট্র্যাকিং ফাংশন (Cache সহ) ---

def load_upload_history():
    global UPLOAD_CACHE
    if UPLOAD_CACHE is not None:
        return UPLOAD_CACHE
    
    try:
        if hf_api:
            file_path = hf_hub_download(
                repo_id=HF_TRACKING_REPO,
                filename="uploaded_files.json",
                repo_type="dataset",
                token=HF_TOKEN
            )
            with open(file_path, 'r', encoding='utf-8') as f:
                UPLOAD_CACHE = json.load(f)
                return UPLOAD_CACHE
    except Exception as e:
        logger.warning(f"HF থেকে লোড করা যায়নি: {e}")
    
    UPLOAD_CACHE = {"files": {}, "total_uploads": 0}
    return UPLOAD_CACHE

def save_upload_history(history):
    global UPLOAD_CACHE
    UPLOAD_CACHE = history
    
    try:
        if hf_api:
            temp_path = "/tmp/uploaded_files.json"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            hf_api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="uploaded_files.json",
                repo_id=HF_TRACKING_REPO,
                repo_type="dataset",
                commit_message=f"Update tracking - {datetime.now().isoformat()}"
            )
            logger.info("✅ Tracking uploaded to Hugging Face")
            return True
    except Exception as e:
        logger.error(f"HF upload failed: {e}")
    return False

def get_file_hash(pdf_bytes):
    return hashlib.md5(pdf_bytes).hexdigest()

def check_file_already_uploaded(file_hash):
    history = load_upload_history()
    if file_hash in history.get('files', {}):
        return {
            'exists': True,
            'date': history['files'][file_hash].get('uploaded_at', 'Unknown'),
            'archive_url': history['files'][file_hash].get('archive_url', ''),
            'filename': history['files'][file_hash].get('filename', '')
        }
    return {'exists': False}

def mark_file_as_uploaded(filename, file_hash, archive_url, pages, vectors, size_mb):
    history = load_upload_history()
    history['files'][file_hash] = {
        'filename': filename,
        'hash': file_hash,
        'uploaded_at': datetime.now().isoformat(),
        'archive_url': archive_url,
        'pages': pages,
        'vectors': vectors,
        'size_mb': size_mb
    }
    history['total_uploads'] = len(history['files'])
    history['last_updated'] = datetime.now().isoformat()
    save_upload_history(history)

# --- ৭. Archive.org ফাংশন (Async wrapper) ---

def upload_to_archive_sync(pdf_bytes, filename, title=""):
    try:
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:10]
        
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
        
        if response[0].status_code == 200:
            logger.info(f"✅ Archive.org upload: {identifier}")
            return {
                'success': True,
                'identifier': identifier,
                'url': f"https://archive.org/details/{identifier}",
                'pdf_url': f"https://archive.org/download/{identifier}/{safe_filename}"
            }
        else:
            logger.error(f"Archive.org upload failed: {response[0].status_code}")
            return {'success': False, 'error': 'Upload failed'}
            
    except Exception as e:
        logger.error(f"Archive.org error: {e}")
        return {'success': False, 'error': str(e)}

async def upload_to_archive(pdf_bytes, filename, title=""):
    """Async wrapper for sync upload"""
    return await asyncio.to_thread(upload_to_archive_sync, pdf_bytes, filename, title)

# --- ৮. Google Drive ফাংশন ---

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

async def download_with_retry(client, url, max_retries=3):
    """Retry logic for network unstable"""
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

# --- ৯. PDF প্রসেসিং ফাংশন ---

def detect_headlines(page_text):
    headlines = []
    lines = page_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_headline = False
        if line.isupper() and len(line) > 3:
            is_headline = True
        elif re.match(r'^[\d\.]+\s+\w+', line):
            is_headline = True
        elif len(line) < 80 and (line.istitle() or line.isupper()):
            is_headline = True
        elif re.match(r'^[=\-]{2,}.*[=\-]{2,}$', line):
            is_headline = True
        if is_headline:
            headlines.append(line)
    return headlines

def extract_paragraphs(page_text):
    raw_paras = re.split(r'\n\s*\n', page_text)
    paragraphs = []
    for para in raw_paras:
        para = para.strip()
        if len(para) > 25:
            para = re.sub(r'\s+', ' ', para)
            paragraphs.append(para)
    return paragraphs

def extract_text_from_pdf_bytes_advanced(pdf_bytes, batch_size=50):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        structured_pages = []
        
        logger.info(f"📚 Total pages: {total_pages}, processing in batches of {batch_size}")
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            logger.info(f"📦 Processing batch: pages {batch_start+1}-{batch_end} of {total_pages}")
            
            for page_num in range(batch_start, batch_end):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue
                
                headlines = detect_headlines(page_text)
                paragraphs = extract_paragraphs(page_text)
                
                structured_pages.append({
                    'page_number': page_num + 1,
                    'headlines': headlines,
                    'paragraphs': paragraphs,
                    'full_text': page_text
                })
                
                if (page_num + 1) % 10 == 0:
                    logger.info(f"   📄 Page {page_num + 1}: {len(headlines)} headlines, {len(paragraphs)} paragraphs")
            
            gc.collect()
            logger.info(f"   🧹 Memory cleared after batch")
        
        logger.info(f"✅ PDF extraction complete: {len(structured_pages)} pages processed")
        return structured_pages
    except Exception as e:
        logger.error(f"PDF এক্সট্র্যাক্ট ত্রুটি: {e}")
        raise

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
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from pypdf import PdfReader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, login, hf_hub_download
from internetarchive import configure, upload
import logging

# ✅ OCR লাইব্রেরি
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available. Install: pytesseract, pdf2image")

# --- ১. লগিং সেটআপ ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ২. কনফিগারেশন (সব Environment Variable থেকে) ---
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

# ✅ ডুপ্লিকেট প্রসেসিং ও রেট লিমিট
processing_tasks = set()
user_limits = {}  # chat_id -> active_tasks_count
MAX_CONCURRENT_PER_USER = 3

# ✅ HF Cache
UPLOAD_CACHE = None

# --- ৩. Hugging Face ট্র্যাকিং সেটআপ ---
hf_api = None
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        hf_api = HfApi(token=HF_TOKEN)
        hf_api.create_repo(repo_id=HF_TRACKING_REPO, repo_type="dataset", exist_ok=True)
        logger.info(f"✅ HF Tracking Repo ready: {HF_TRACKING_REPO}")
    except Exception as e:
        logger.error(f"❌ HF Tracking setup failed: {e}")

# --- ৪. Archive.org সেটআপ ---
if IA_EMAIL and IA_PASSWORD:
    try:
        configure(IA_EMAIL, IA_PASSWORD)
        logger.info("✅ Archive.org configured")
    except Exception as e:
        logger.error(f"❌ Archive.org setup failed: {e}")

# --- ৫. Pinecone ও এম্বেডিং মডেল সেটআপ ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Pinecone ও এম্বেডিং মডেল লোড হয়েছে")
except Exception as e:
    logger.error(f"❌ Pinecone সেটআপ ত্রুটি: {e}")
    index = None
    embedding_model = None

# --- ৬. HF ট্র্যাকিং ফাংশন (Cache সহ) ---

def load_upload_history():
    global UPLOAD_CACHE
    if UPLOAD_CACHE is not None:
        return UPLOAD_CACHE
    
    try:
        if hf_api:
            file_path = hf_hub_download(
                repo_id=HF_TRACKING_REPO,
                filename="uploaded_files.json",
                repo_type="dataset",
                token=HF_TOKEN
            )
            with open(file_path, 'r', encoding='utf-8') as f:
                UPLOAD_CACHE = json.load(f)
                return UPLOAD_CACHE
    except Exception as e:
        logger.warning(f"HF থেকে লোড করা যায়নি: {e}")
    
    UPLOAD_CACHE = {"files": {}, "total_uploads": 0}
    return UPLOAD_CACHE

def save_upload_history(history):
    global UPLOAD_CACHE
    UPLOAD_CACHE = history
    
    try:
        if hf_api:
            temp_path = "/tmp/uploaded_files.json"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            hf_api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="uploaded_files.json",
                repo_id=HF_TRACKING_REPO,
                repo_type="dataset",
                commit_message=f"Update tracking - {datetime.now().isoformat()}"
            )
            logger.info("✅ Tracking uploaded to Hugging Face")
            return True
    except Exception as e:
        logger.error(f"HF upload failed: {e}")
    return False

def get_file_hash(pdf_bytes):
    return hashlib.md5(pdf_bytes).hexdigest()

def check_file_already_uploaded(file_hash):
    history = load_upload_history()
    if file_hash in history.get('files', {}):
        return {
            'exists': True,
            'date': history['files'][file_hash].get('uploaded_at', 'Unknown'),
            'archive_url': history['files'][file_hash].get('archive_url', ''),
            'filename': history['files'][file_hash].get('filename', '')
        }
    return {'exists': False}

def mark_file_as_uploaded(filename, file_hash, archive_url, pages, vectors, size_mb):
    history = load_upload_history()
    history['files'][file_hash] = {
        'filename': filename,
        'hash': file_hash,
        'uploaded_at': datetime.now().isoformat(),
        'archive_url': archive_url,
        'pages': pages,
        'vectors': vectors,
        'size_mb': size_mb
    }
    history['total_uploads'] = len(history['files'])
    history['last_updated'] = datetime.now().isoformat()
    save_upload_history(history)

# --- ৭. Archive.org ফাংশন (Async wrapper) ---

def upload_to_archive_sync(pdf_bytes, filename, title=""):
    try:
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:10]
        
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
        
        if response[0].status_code == 200:
            logger.info(f"✅ Archive.org upload: {identifier}")
            return {
                'success': True,
                'identifier': identifier,
                'url': f"https://archive.org/details/{identifier}",
                'pdf_url': f"https://archive.org/download/{identifier}/{safe_filename}"
            }
        else:
            logger.error(f"Archive.org upload failed: {response[0].status_code}")
            return {'success': False, 'error': 'Upload failed'}
            
    except Exception as e:
        logger.error(f"Archive.org error: {e}")
        return {'success': False, 'error': str(e)}

async def upload_to_archive(pdf_bytes, filename, title=""):
    """Async wrapper for sync upload"""
    return await asyncio.to_thread(upload_to_archive_sync, pdf_bytes, filename, title)

# --- ৮. Google Drive ফাংশন ---

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

async def download_with_retry(client, url, max_retries=3):
    """Retry logic for network unstable"""
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

# --- ৯. PDF প্রসেসিং ফাংশন ---

def detect_headlines(page_text):
    headlines = []
    lines = page_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_headline = False
        if line.isupper() and len(line) > 3:
            is_headline = True
        elif re.match(r'^[\d\.]+\s+\w+', line):
            is_headline = True
        elif len(line) < 80 and (line.istitle() or line.isupper()):
            is_headline = True
        elif re.match(r'^[=\-]{2,}.*[=\-]{2,}$', line):
            is_headline = True
        if is_headline:
            headlines.append(line)
    return headlines

def extract_paragraphs(page_text):
    raw_paras = re.split(r'\n\s*\n', page_text)
    paragraphs = []
    for para in raw_paras:
        para = para.strip()
        if len(para) > 25:
            para = re.sub(r'\s+', ' ', para)
            paragraphs.append(para)
    return paragraphs

def extract_text_from_pdf_bytes_advanced(pdf_bytes, batch_size=50):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        structured_pages = []
        
        logger.info(f"📚 Total pages: {total_pages}, processing in batches of {batch_size}")
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            logger.info(f"📦 Processing batch: pages {batch_start+1}-{batch_end} of {total_pages}")
            
            for page_num in range(batch_start, batch_end):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue
                
                headlines = detect_headlines(page_text)
                paragraphs = extract_paragraphs(page_text)
                
                structured_pages.append({
                    'page_number': page_num + 1,
                    'headlines': headlines,
                    'paragraphs': paragraphs,
                    'full_text': page_text
                })
                
                if (page_num + 1) % 10 == 0:
                    logger.info(f"   📄 Page {page_num + 1}: {len(headlines)} headlines, {len(paragraphs)} paragraphs")
            
            gc.collect()
            logger.info(f"   🧹 Memory cleared after batch")
        
        logger.info(f"✅ PDF extraction complete: {len(structured_pages)} pages processed")
        return structured_pages
    except Exception as e:
        logger.error(f"PDF এক্সট্র্যাক্ট ত্রুটি: {e}")
        raise

def extract_text_from_pdf_bytes_ocr(pdf_bytes, batch_size=10):
    if not OCR_AVAILABLE:
        logger.error("❌ OCR libraries not available")
        return []
    
    try:
        # প্রথমে total pages বের করতে একটি ইমেজ লোড
        test_images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
        if not test_images:
            return []
        
        # Total pages বের করতে PdfReader ব্যবহার
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        structured_pages = []
        
        logger.info(f"📚 Total pages: {total_pages}, processing with OCR in batches of {batch_size}")
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            logger.info(f"🖼️ Processing OCR batch: pages {batch_start+1}-{batch_end} of {total_pages}")
            
            # ✅ শুধু এই ব্যাচের পৃষ্ঠা লোড (Memory safe)
            images = convert_from_bytes(
                pdf_bytes, 
                dpi=150, 
                first_page=batch_start + 1, 
                last_page=batch_end
            )
            
            for idx, image in enumerate(images):
                page_num = batch_start + idx + 1
                
                try:
                    page_text = pytesseract.image_to_string(image, lang='ben+eng')
                except:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                
                if not page_text or not page_text.strip():
                    continue
                
                headlines = detect_headlines(page_text)
                paragraphs = extract_paragraphs(page_text)
                
                structured_pages.append({
                    'page_number': page_num,
                    'headlines': headlines,
                    'paragraphs': paragraphs,
                    'full_text': page_text
                })
                
                if page_num % 5 == 0:
                    logger.info(f"   📄 Page {page_num}: {len(headlines)} headlines, {len(paragraphs)} paragraphs")
            
            # ইমেজ ক্লিয়ার
            del images
            gc.collect()
            logger.info(f"   🧹 Memory cleared after OCR batch")
        
        logger.info(f"✅ OCR extraction complete: {len(structured_pages)} pages processed")
        return structured_pages
        
    except Exception as e:
        logger.error(f"OCR এক্সট্র্যাক্ট ত্রুটি: {e}")
        raise

def extract_text_from_pdf_bytes_auto(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) == 0:
            return []
        
        first_page_text = reader.pages[0].extract_text() if len(reader.pages) > 0 else ""
        
        if len(first_page_text.strip()) < 50:
            logger.info("📷 Detected scanned PDF, using OCR...")
            if OCR_AVAILABLE:
                return extract_text_from_pdf_bytes_ocr(pdf_bytes)
            else:
                logger.warning("⚠️ OCR not available, falling back to standard extraction")
                return extract_text_from_pdf_bytes_advanced(pdf_bytes)
        else:
            logger.info("📄 Detected text-based PDF, using standard extraction...")
            return extract_text_from_pdf_bytes_advanced(pdf_bytes)
            
    except Exception as e:
        logger.warning(f"Auto-detect failed, falling back to standard: {e}")
        return extract_text_from_pdf_bytes_advanced(pdf_bytes)

def create_structured_chunks(structured_pages, filename):
    chunks = []
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    for page in structured_pages:
        page_num = page['page_number']
        headlines = page['headlines']
        paragraphs = page['paragraphs']
        
        for headline in headlines:
            chunks.append({
                'text': headline,
                'metadata': {'page': page_num, 'type': 'headline', 'headline': headline[:100]}
            })
        
        for para_num, para_text in enumerate(paragraphs, 1):
            related_headline = headlines[0] if headlines else "No headline"
            chunks.append({
                'text': para_text,
                'metadata': {
                    'page': page_num, 'type': 'paragraph', 'para_number': para_num,
                    'headline': related_headline[:100], 'total_paras': len(paragraphs)
                }
            })
        
        if page['full_text'].strip():
            chunks.append({
                'text': page['full_text'][:2000],
                'metadata': {
                    'page': page_num, 'type': 'full_page',
                    'headline': headlines[0][:100] if headlines else "No headline"
                }
            })
    
    return chunks

def save_structured_to_pinecone(filename, chunks):
    if index is None or embedding_model is None:
        raise Exception("Pinecone বা এম্বেডিং মডেল লোড হয়নি")
    
    vectors = []
    batch_size = 50
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    for i, chunk_data in enumerate(chunks):
        chunk_text = chunk_data['text']
        metadata = chunk_data['metadata']
        embedding = embedding_model.encode(chunk_text).tolist()
        
        full_metadata = {
            "filename": filename, "chunk_index": i, "text": chunk_text[:1000], **metadata
        }
        
        vectors.append({
            "id": f"{file_hash}_{i}",
            "values": embedding,
            "metadata": full_metadata
        })
        
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            logger.info(f"   📤 Uploaded {len(vectors)} vectors to Pinecone")
            vectors = []
            gc.collect()
    
    if vectors:
        index.upsert(vectors=vectors)
        logger.info(f"   📤 Uploaded final {len(vectors)} vectors to Pinecone")
    
    return len(chunks)

def search_in_pinecone_advanced(query, top_k=5):
    if index is None or embedding_model is None:
        return []
    
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        
        chunks = []
        for match in results.matches:
            if match.score > 0.05:
                metadata = match.metadata
                chunks.append({
                    "text": metadata.get("text", ""),
                    "filename": metadata.get("filename", ""),
                    "page": metadata.get("page", "N/A"),
                    "headline": metadata.get("headline", "N/A"),
                    "para_number": metadata.get("para_number", "N/A"),
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
    
    headlines = [r for r in results if r.get('type') == 'headline']
    paragraphs = [r for r in results if r.get('type') == 'paragraph']
    
    if headlines:
        formatted += "📌 প্রাসঙ্গিক হেডলাইন:\n"
        for h in headlines[:3]:
            formatted += f"• [পৃষ্ঠা {h['page']}] {h['text'][:100]}...\n"
        formatted += "\n"
    
    if paragraphs:
        formatted += "📝 প্রাসঙ্গিক অংশ:\n"
        for p in paragraphs[:2]:
            formatted += f"\n📍 পৃষ্ঠা {p['page']}"
            if p.get('headline') and p['headline'] != 'N/A':
                formatted += f" | {p['headline'][:40]}"
            if p.get('para_number') and p['para_number'] != 'N/A':
                formatted += f" | প্যারা {p['para_number']}"
            formatted += f"\n{p['text'][:300]}...\n"
    
    formatted += f"\n---\n📚 সোর্স: {results[0].get('filename', 'Unknown')}"
    
    # ✅ Telegram limit (4096 chars)
    if len(formatted) > 4000:
        formatted = formatted[:4000] + "..."
    
    return formatted

# --- ১০. Telegram বট হ্যান্ডলার ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Quran PDF Bot\n\n"
        "/help - সকল কমান্ড দেখুন\n"
        "/status - সিস্টেম স্ট্যাটাস\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n\n"
        "✨ Google Drive লিংক দিন বড় PDF আপলোড করতে!"
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
        "প্রশ্ন: সরাসরি প্রশ্ন লিখুন"
    )
    await update.message.reply_text(help_text)

async def handle_drive_folder(update: Update, context: ContextTypes.DEFAULT_TYPE, folder_id):
    status_msg = await update.message.reply_text("📁 ফোল্ডার স্ক্যান করা হচ্ছে...")
    # ... (ফোল্ডার হ্যান্ডলিং কোড অপরিবর্তিত)

async def process_pdf_background_safe(folder_id, chat_id, bot):
    """নিরাপদ ব্যাকগ্রাউন্ড PDF প্রসেসিং (Timeout + Retry সহ)"""
    try:
        # ✅ ৩০ মিনিট টাইমআউট
        await asyncio.wait_for(
            _process_pdf_internal(folder_id, chat_id, bot),
            timeout=1800  # 30 minutes
        )
    except asyncio.TimeoutError:
        await bot.send_message(chat_id, "⏰ প্রক্রিয়াকরণ সময়সীমা অতিক্রম করেছে (৩০ মিনিট)। দয়া করে ছোট PDF আপলোড করুন।")
    except Exception as e:
        logger.error(f"❌ Background processing error: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ প্রক্রিয়াকরণে ত্রুটি: {str(e)}")
    finally:
        processing_tasks.discard(folder_id)
        if chat_id in user_limits:
            user_limits[chat_id] = max(0, user_limits.get(chat_id, 1) - 1)

async def _process_pdf_internal(folder_id, chat_id, bot):
    """Actual PDF processing"""
    download_url = f"https://drive.google.com/uc?export=download&id={folder_id}"
    
    # ✅ Retry logic সহ ডাউনলোড
    async with httpx.AsyncClient(timeout=120) as client:
        response = await download_with_retry(client, download_url)
    
    if response.status_code != 200:
        await bot.send_message(chat_id, f"❌ ডাউনলোড ব্যর্থ: {response.status_code}")
        return
    
    pdf_bytes = response.content
    file_hash = get_file_hash(pdf_bytes)
    
    duplicate = check_file_already_uploaded(file_hash)
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
    
    # Archive.org আপলোড (Async)
    archive_result = await upload_to_archive(pdf_bytes, filename)
    archive_url = archive_result.get('url') if archive_result['success'] else None
    
    # PDF প্রসেসিং (Sync, কিন্তু দ্রুত)
    structured_pages = extract_text_from_pdf_bytes_auto(pdf_bytes)
    chunks = create_structured_chunks(structured_pages, filename)
    vector_count = save_structured_to_pinecone(filename, chunks)
    
    total_pages = len(structured_pages)
    total_headlines = sum(len(p['headlines']) for p in structured_pages)
    total_paras = sum(len(p['paragraphs']) for p in structured_pages)
    
    mark_file_as_uploaded(filename, file_hash, archive_url, total_pages, vector_count, size_mb)
    
    final_msg = f"✅ আপনার PDF সফলভাবে সংরক্ষিত হয়েছে!\n\n"
    final_msg += f"📄 পৃষ্ঠা: {total_pages}\n"
    final_msg += f"📌 হেডলাইন: {total_headlines}\n"
    final_msg += f"📝 প্যারাগ্রাফ: {total_paras}\n"
    final_msg += f"🗄️ ভেক্টর: {vector_count}\n"
    final_msg += f"📊 আকার: {size_mb:.2f} MB\n"
    
    if archive_url:
        final_msg += f"\n📚 Archive.org: {archive_url}\n\n"
    
    final_msg += "🎉 এখন আপনি এই PDF সম্পর্কে প্রশ্ন করতে পারেন!"
    
    await bot.send_message(chat_id, final_msg)

async def handle_drive_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text
    logger.info(f"🚀 handle_drive_link started with URL: {url}")
    
    if 'drive.google.com' not in url:
        return
    
    folder_id, url_type = extract_folder_id_from_url(url)
    
    if not folder_id:
        await update.message.reply_text("❌ অবৈধ Google Drive লিংক।")
        return
    
    if url_type == 'folder':
        await handle_drive_folder(update, context, folder_id)
        return
    
    chat_id = update.effective_chat.id
    
    # ✅ ডুপ্লিকেট প্রসেসিং চেক
    if folder_id in processing_tasks:
        await update.message.reply_text("⚠️ এই PDFটি ইতিমধ্যে প্রক্রিয়াকরণ হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন...")
        return
    
    # ✅ রেট লিমিট চেক
    active_tasks = user_limits.get(chat_id, 0)
    if active_tasks >= MAX_CONCURRENT_PER_USER:
        await update.message.reply_text(f"⚠️ আপনি ইতিমধ্যে {active_tasks}টি PDF প্রক্রিয়াকরণ করছেন। অনুগ্রহ করে আগেরগুলো শেষ হওয়ার জন্য অপেক্ষা করুন।")
        return
    
    processing_tasks.add(folder_id)
    user_limits[chat_id] = active_tasks + 1
    
    await update.message.reply_text(
        "📥 আপনার PDF প্রক্রিয়াকরণ শুরু হয়েছে!\n\n"
        "⏳ এটি স্ক্যান করা PDF হলে ২০-৩০ মিনিট সময় লাগতে পারে।\n"
        "✅ প্রক্রিয়া শেষে আপনাকে জানানো হবে।\n\n"
        "🙏 অনুগ্রহ করে অপেক্ষা করুন..."
    )
    
    # ✅ নিরাপদ ব্যাকগ্রাউন্ড প্রসেসিং
    asyncio.create_task(process_pdf_background_safe(folder_id, chat_id, context.bot))

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    
    if 'drive.google.com' in user_question:
        await handle_drive_link(update, context)
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        results = search_in_pinecone_advanced(user_question, top_k=5)
        formatted_answer = format_search_results(results, user_question)
        await update.message.reply_text(formatted_answer)
    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def list_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if index is None:
            await update.message.reply_text("❌ Pinecone সংযুক্ত নয়")
            return
        
        results = index.query(vector=[0.1]*384, top_k=1000, include_metadata=True)
        
        file_stats = {}
        for match in results.matches:
            if match.metadata and 'filename' in match.metadata:
                filename = match.metadata['filename']
                if filename not in file_stats:
                    file_stats[filename] = {'pages': set(), 'vectors': 0}
                page = match.metadata.get('page')
                if page:
                    file_stats[filename]['pages'].add(page)
                file_stats[filename]['vectors'] += 1
        
        if file_stats:
            file_list = ""
            for filename, stats in file_stats.items():
                file_list += f"\n📁 {filename}\n"
                file_list += f"   📄 পৃষ্ঠা: {len(stats['pages'])}\n"
                file_list += f"   🗄️ ভেক্টর: {stats['vectors']}\n"
            await update.message.reply_text(f"সংরক্ষিত PDF:\n{file_list}")
        else:
            await update.message.reply_text("ℹ️ এখনো কোনো PDF সংরক্ষিত হয়নি।")
    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pc_status = "✅" if index is not None else "❌"
    hf_status = "✅" if hf_api is not None else "❌"
    ia_status = "✅" if IA_EMAIL and IA_PASSWORD else "❌"
    ocr_status = "✅" if OCR_AVAILABLE else "❌"
    
    status_text = (
        f"📊 সিস্টেম স্ট্যাটাস\n\n"
        f"🗄️ Pinecone: {pc_status}\n"
        f"📚 Archive.org: {ia_status}\n"
        f"🔍 HF Tracking: {hf_status}\n"
        f"📷 OCR: {ocr_status}\n"
        f"📁 ইনডেক্স: {PINECONE_INDEX_NAME}"
    )
    await update.message.reply_text(status_text)

# --- ১১. FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    request = HTTPXRequest(connection_pool_size=10, read_timeout=120, write_timeout=120)
    bot = Bot(token=TELEGRAM_TOKEN, request=request)
    
    ptb_app = Application.builder().bot(bot).build()
    
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("help", help_command))
    ptb_app.add_handler(CommandHandler("list", list_files))
    ptb_app.add_handler(CommandHandler("status", status))
    ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_question))
    
    await ptb_app.initialize()
    app.state.ptb_app = ptb_app
    app.state.bot = bot
    
    webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
    await bot.set_webhook(url=webhook_url, secret_token=SECRET_TOKEN)
    logger.info(f"✅ Webhook set to: {webhook_url}")
    
    yield
    await bot.delete_webhook()
    await ptb_app.shutdown()

app = FastAPI(lifespan=lifespan)

# --- ১২. রুট এন্ডপয়েন্ট ---
@app.get("/")
async def root():
    return {"status": "ok", "service": "Quran PDF Bot"}

@app.get("/healthcheck")
async def health():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    # ✅ Webhook Security Check
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != SECRET_TOKEN:
        logger.warning("❌ Invalid secret token")
        return Response(status_code=403)
    
    ptb_app = request.app.state.ptb_app
    bot = request.app.state.bot
    
    data = await request.json()
    update = Update.de_json(data, bot)
    
    await ptb_app.process_update(update)
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

def extract_text_from_pdf_bytes_auto(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) == 0:
            return []
        
        first_page_text = reader.pages[0].extract_text() if len(reader.pages) > 0 else ""
        
        if len(first_page_text.strip()) < 50:
            logger.info("📷 Detected scanned PDF, using OCR...")
            if OCR_AVAILABLE:
                return extract_text_from_pdf_bytes_ocr(pdf_bytes)
            else:
                logger.warning("⚠️ OCR not available, falling back to standard extraction")
                return extract_text_from_pdf_bytes_advanced(pdf_bytes)
        else:
            logger.info("📄 Detected text-based PDF, using standard extraction...")
            return extract_text_from_pdf_bytes_advanced(pdf_bytes)
            
    except Exception as e:
        logger.warning(f"Auto-detect failed, falling back to standard: {e}")
        return extract_text_from_pdf_bytes_advanced(pdf_bytes)

def create_structured_chunks(structured_pages, filename):
    chunks = []
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    for page in structured_pages:
        page_num = page['page_number']
        headlines = page['headlines']
        paragraphs = page['paragraphs']
        
        for headline in headlines:
            chunks.append({
                'text': headline,
                'metadata': {'page': page_num, 'type': 'headline', 'headline': headline[:100]}
            })
        
        for para_num, para_text in enumerate(paragraphs, 1):
            related_headline = headlines[0] if headlines else "No headline"
            chunks.append({
                'text': para_text,
                'metadata': {
                    'page': page_num, 'type': 'paragraph', 'para_number': para_num,
                    'headline': related_headline[:100], 'total_paras': len(paragraphs)
                }
            })
        
        if page['full_text'].strip():
            chunks.append({
                'text': page['full_text'][:2000],
                'metadata': {
                    'page': page_num, 'type': 'full_page',
                    'headline': headlines[0][:100] if headlines else "No headline"
                }
            })
    
    return chunks

def save_structured_to_pinecone(filename, chunks):
    if index is None or embedding_model is None:
        raise Exception("Pinecone বা এম্বেডিং মডেল লোড হয়নি")
    
    vectors = []
    batch_size = 50
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    for i, chunk_data in enumerate(chunks):
        chunk_text = chunk_data['text']
        metadata = chunk_data['metadata']
        embedding = embedding_model.encode(chunk_text).tolist()
        
        full_metadata = {
            "filename": filename, "chunk_index": i, "text": chunk_text[:1000], **metadata
        }
        
        vectors.append({
            "id": f"{file_hash}_{i}",
            "values": embedding,
            "metadata": full_metadata
        })
        
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            logger.info(f"   📤 Uploaded {len(vectors)} vectors to Pinecone")
            vectors = []
            gc.collect()
    
    if vectors:
        index.upsert(vectors=vectors)
        logger.info(f"   📤 Uploaded final {len(vectors)} vectors to Pinecone")
    
    return len(chunks)

def search_in_pinecone_advanced(query, top_k=5):
    if index is None or embedding_model is None:
        return []
    
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        
        chunks = []
        for match in results.matches:
            if match.score > 0.05:
                metadata = match.metadata
                chunks.append({
                    "text": metadata.get("text", ""),
                    "filename": metadata.get("filename", ""),
                    "page": metadata.get("page", "N/A"),
                    "headline": metadata.get("headline", "N/A"),
                    "para_number": metadata.get("para_number", "N/A"),
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
    
    headlines = [r for r in results if r.get('type') == 'headline']
    paragraphs = [r for r in results if r.get('type') == 'paragraph']
    
    if headlines:
        formatted += "📌 প্রাসঙ্গিক হেডলাইন:\n"
        for h in headlines[:3]:
            formatted += f"• [পৃষ্ঠা {h['page']}] {h['text'][:100]}...\n"
        formatted += "\n"
    
    if paragraphs:
        formatted += "📝 প্রাসঙ্গিক অংশ:\n"
        for p in paragraphs[:2]:
            formatted += f"\n📍 পৃষ্ঠা {p['page']}"
            if p.get('headline') and p['headline'] != 'N/A':
                formatted += f" | {p['headline'][:40]}"
            if p.get('para_number') and p['para_number'] != 'N/A':
                formatted += f" | প্যারা {p['para_number']}"
            formatted += f"\n{p['text'][:300]}...\n"
    
    formatted += f"\n---\n📚 সোর্স: {results[0].get('filename', 'Unknown')}"
    
    # ✅ Telegram limit (4096 chars)
    if len(formatted) > 4000:
        formatted = formatted[:4000] + "..."
    
    return formatted

# --- ১০. Telegram বট হ্যান্ডলার ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Quran PDF Bot\n\n"
        "/help - সকল কমান্ড দেখুন\n"
        "/status - সিস্টেম স্ট্যাটাস\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n\n"
        "✨ Google Drive লিংক দিন বড় PDF আপলোড করতে!"
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
        "প্রশ্ন: সরাসরি প্রশ্ন লিখুন"
    )
    await update.message.reply_text(help_text)

async def handle_drive_folder(update: Update, context: ContextTypes.DEFAULT_TYPE, folder_id):
    status_msg = await update.message.reply_text("📁 ফোল্ডার স্ক্যান করা হচ্ছে...")
    # ... (ফোল্ডার হ্যান্ডলিং কোড অপরিবর্তিত)

async def process_pdf_background_safe(folder_id, chat_id, bot):
    """নিরাপদ ব্যাকগ্রাউন্ড PDF প্রসেসিং (Timeout + Retry সহ)"""
    try:
        # ✅ ৩০ মিনিট টাইমআউট
        await asyncio.wait_for(
            _process_pdf_internal(folder_id, chat_id, bot),
            timeout=1800  # 30 minutes
        )
    except asyncio.TimeoutError:
        await bot.send_message(chat_id, "⏰ প্রক্রিয়াকরণ সময়সীমা অতিক্রম করেছে (৩০ মিনিট)। দয়া করে ছোট PDF আপলোড করুন।")
    except Exception as e:
        logger.error(f"❌ Background processing error: {e}", exc_info=True)
        await bot.send_message(chat_id, f"❌ প্রক্রিয়াকরণে ত্রুটি: {str(e)}")
    finally:
        processing_tasks.discard(folder_id)
        if chat_id in user_limits:
            user_limits[chat_id] = max(0, user_limits.get(chat_id, 1) - 1)

async def _process_pdf_internal(folder_id, chat_id, bot):
    """Actual PDF processing"""
    download_url = f"https://drive.google.com/uc?export=download&id={folder_id}"
    
    # ✅ Retry logic সহ ডাউনলোড
    async with httpx.AsyncClient(timeout=120) as client:
        response = await download_with_retry(client, download_url)
    
    if response.status_code != 200:
        await bot.send_message(chat_id, f"❌ ডাউনলোড ব্যর্থ: {response.status_code}")
        return
    
    pdf_bytes = response.content
    file_hash = get_file_hash(pdf_bytes)
    
    duplicate = check_file_already_uploaded(file_hash)
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
    
    # Archive.org আপলোড (Async)
    archive_result = await upload_to_archive(pdf_bytes, filename)
    archive_url = archive_result.get('url') if archive_result['success'] else None
    
    # PDF প্রসেসিং (Sync, কিন্তু দ্রুত)
    structured_pages = extract_text_from_pdf_bytes_auto(pdf_bytes)
    chunks = create_structured_chunks(structured_pages, filename)
    vector_count = save_structured_to_pinecone(filename, chunks)
    
    total_pages = len(structured_pages)
    total_headlines = sum(len(p['headlines']) for p in structured_pages)
    total_paras = sum(len(p['paragraphs']) for p in structured_pages)
    
    mark_file_as_uploaded(filename, file_hash, archive_url, total_pages, vector_count, size_mb)
    
    final_msg = f"✅ আপনার PDF সফলভাবে সংরক্ষিত হয়েছে!\n\n"
    final_msg += f"📄 পৃষ্ঠা: {total_pages}\n"
    final_msg += f"📌 হেডলাইন: {total_headlines}\n"
    final_msg += f"📝 প্যারাগ্রাফ: {total_paras}\n"
    final_msg += f"🗄️ ভেক্টর: {vector_count}\n"
    final_msg += f"📊 আকার: {size_mb:.2f} MB\n"
    
    if archive_url:
        final_msg += f"\n📚 Archive.org: {archive_url}\n\n"
    
    final_msg += "🎉 এখন আপনি এই PDF সম্পর্কে প্রশ্ন করতে পারেন!"
    
    await bot.send_message(chat_id, final_msg)

async def handle_drive_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text
    logger.info(f"🚀 handle_drive_link started with URL: {url}")
    
    if 'drive.google.com' not in url:
        return
    
    folder_id, url_type = extract_folder_id_from_url(url)
    
    if not folder_id:
        await update.message.reply_text("❌ অবৈধ Google Drive লিংক।")
        return
    
    if url_type == 'folder':
        await handle_drive_folder(update, context, folder_id)
        return
    
    chat_id = update.effective_chat.id
    
    # ✅ ডুপ্লিকেট প্রসেসিং চেক
    if folder_id in processing_tasks:
        await update.message.reply_text("⚠️ এই PDFটি ইতিমধ্যে প্রক্রিয়াকরণ হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন...")
        return
    
    # ✅ রেট লিমিট চেক
    active_tasks = user_limits.get(chat_id, 0)
    if active_tasks >= MAX_CONCURRENT_PER_USER:
        await update.message.reply_text(f"⚠️ আপনি ইতিমধ্যে {active_tasks}টি PDF প্রক্রিয়াকরণ করছেন। অনুগ্রহ করে আগেরগুলো শেষ হওয়ার জন্য অপেক্ষা করুন।")
        return
    
    processing_tasks.add(folder_id)
    user_limits[chat_id] = active_tasks + 1
    
    await update.message.reply_text(
        "📥 আপনার PDF প্রক্রিয়াকরণ শুরু হয়েছে!\n\n"
        "⏳ এটি স্ক্যান করা PDF হলে ২০-৩০ মিনিট সময় লাগতে পারে।\n"
        "✅ প্রক্রিয়া শেষে আপনাকে জানানো হবে।\n\n"
        "🙏 অনুগ্রহ করে অপেক্ষা করুন..."
    )
    
    # ✅ নিরাপদ ব্যাকগ্রাউন্ড প্রসেসিং
    asyncio.create_task(process_pdf_background_safe(folder_id, chat_id, context.bot))

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    
    if 'drive.google.com' in user_question:
        await handle_drive_link(update, context)
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        results = search_in_pinecone_advanced(user_question, top_k=5)
        formatted_answer = format_search_results(results, user_question)
        await update.message.reply_text(formatted_answer)
    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def list_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if index is None:
            await update.message.reply_text("❌ Pinecone সংযুক্ত নয়")
            return
        
        results = index.query(vector=[0.1]*384, top_k=1000, include_metadata=True)
        
        file_stats = {}
        for match in results.matches:
            if match.metadata and 'filename' in match.metadata:
                filename = match.metadata['filename']
                if filename not in file_stats:
                    file_stats[filename] = {'pages': set(), 'vectors': 0}
                page = match.metadata.get('page')
                if page:
                    file_stats[filename]['pages'].add(page)
                file_stats[filename]['vectors'] += 1
        
        if file_stats:
            file_list = ""
            for filename, stats in file_stats.items():
                file_list += f"\n📁 {filename}\n"
                file_list += f"   📄 পৃষ্ঠা: {len(stats['pages'])}\n"
                file_list += f"   🗄️ ভেক্টর: {stats['vectors']}\n"
            await update.message.reply_text(f"সংরক্ষিত PDF:\n{file_list}")
        else:
            await update.message.reply_text("ℹ️ এখনো কোনো PDF সংরক্ষিত হয়নি।")
    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pc_status = "✅" if index is not None else "❌"
    hf_status = "✅" if hf_api is not None else "❌"
    ia_status = "✅" if IA_EMAIL and IA_PASSWORD else "❌"
    ocr_status = "✅" if OCR_AVAILABLE else "❌"
    
    status_text = (
        f"📊 সিস্টেম স্ট্যাটাস\n\n"
        f"🗄️ Pinecone: {pc_status}\n"
        f"📚 Archive.org: {ia_status}\n"
        f"🔍 HF Tracking: {hf_status}\n"
        f"📷 OCR: {ocr_status}\n"
        f"📁 ইনডেক্স: {PINECONE_INDEX_NAME}"
    )
    await update.message.reply_text(status_text)

# --- ১১. FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    request = HTTPXRequest(connection_pool_size=10, read_timeout=120, write_timeout=120)
    bot = Bot(token=TELEGRAM_TOKEN, request=request)
    
    ptb_app = Application.builder().bot(bot).build()
    
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("help", help_command))
    ptb_app.add_handler(CommandHandler("list", list_files))
    ptb_app.add_handler(CommandHandler("status", status))
    ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_question))
    
    await ptb_app.initialize()
    app.state.ptb_app = ptb_app
    app.state.bot = bot
    
    webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
    await bot.set_webhook(url=webhook_url, secret_token=SECRET_TOKEN)
    logger.info(f"✅ Webhook set to: {webhook_url}")
    
    yield
    await bot.delete_webhook()
    await ptb_app.shutdown()

app = FastAPI(lifespan=lifespan)

# --- ১২. রুট এন্ডপয়েন্ট ---
@app.get("/")
async def root():
    return {"status": "ok", "service": "Quran PDF Bot"}

@app.get("/healthcheck")
async def health():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    # ✅ Webhook Security Check
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != SECRET_TOKEN:
        logger.warning("❌ Invalid secret token")
        return Response(status_code=403)
    
    ptb_app = request.app.state.ptb_app
    bot = request.app.state.bot
    
    data = await request.json()
    update = Update.de_json(data, bot)
    
    await ptb_app.process_update(update)
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
