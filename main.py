#!/usr/bin/env python3
"""
Internet Archive PDF Processor - GitHub Actions Cron Job
হায়ারার্কি: অ্যাকাউন্ট → ফোল্ডার → আইটেম → PDF
"""

import os
import asyncio
import io
import re
import json
import hashlib
import gc
import uuid
import httpx
from datetime import datetime
from pathlib import Path
from pypdf import PdfReader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, login, hf_hub_download
import logging

# ✅ নতুন: Internet Archive লাইব্রেরি (লগইনের জন্য)
try:
    from internetarchive import get_session
    IA_LIB_AVAILABLE = True
except ImportError:
    IA_LIB_AVAILABLE = False
    print("⚠️ internetarchive library not available. Install: pip install internetarchive")

# ✅ OCR লাইব্রেরি
try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available")

# --- লগিং সেটআপ ---
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/processor.log')
    ]
)
logger = logging.getLogger(__name__)

# --- কনফিগারেশন ---
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quranqpf")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IA_ACCOUNT_ID = os.environ.get("IA_ACCOUNT_ID", "ahashan_ahmed185")

# ✅ নতুন: ইমেইল ও পাসওয়ার্ড এনভায়রনমেন্ট ভেরিয়েবল
IA_EMAIL = os.environ.get("IA_EMAIL")
IA_PASSWORD = os.environ.get("IA_PASSWORD")

# HF ট্র্যাকিং রেপো
HF_TRACKING_REPO = "ahashanahmed/quran-bot-tracking"

# ✅ চেকপয়েন্ট ফাইল
CHECKPOINT_DIR = Path("/tmp/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
PAGE_CHECKPOINT_FILE = CHECKPOINT_DIR / "page_checkpoint.json"

# ✅ OCR কনফিগারেশন
OCR_DPI = 200
OCR_LANG_PRIMARY = "ben+ara"
OCR_TIMEOUT = 10

# ✅ ব্যাচ সাইজ
PINECONE_BATCH_SIZE = 10

# ✅ গ্লোবাল ভেরিয়েবল
index = None
embedding_model = None
hf_api = None
pc = None
ia_session = None  # ✅ নতুন: Internet Archive সেশন

# --- চেকপয়েন্ট সিস্টেম ---
def load_page_checkpoint():
    """পৃষ্ঠা চেকপয়েন্ট লোড"""
    try:
        if PAGE_CHECKPOINT_FILE.exists():
            with open(PAGE_CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_page_checkpoint(checkpoint):
    """পৃষ্ঠা চেকপয়েন্ট সংরক্ষণ"""
    try:
        with open(PAGE_CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except:
        pass

def get_file_progress(file_hash):
    """নির্দিষ্ট ফাইলের অগ্রগতি"""
    checkpoint = load_page_checkpoint()
    return checkpoint.get(file_hash, {'last_page': 0, 'completed': False, 'total_pages': 0})

def update_file_progress(file_hash, page_num, total_pages=None, completed=False):
    """ফাইলের অগ্রগতি আপডেট"""
    checkpoint = load_page_checkpoint()
    existing = checkpoint.get(file_hash, {})

    checkpoint[file_hash] = {
        'last_page': page_num,
        'completed': completed,
        'total_pages': total_pages or existing.get('total_pages', 0),
        'updated_at': datetime.now().isoformat()
    }
    save_page_checkpoint(checkpoint)
    logger.info(f"   💾 Checkpoint saved: page {page_num}")

def is_file_completed(file_hash):
    """ফাইল সম্পূর্ণ প্রক্রিয়াকৃত কিনা"""
    progress = get_file_progress(file_hash)
    return progress.get('completed', False)

# --- Internet Archive লগইন (ইমেইল/পাসওয়ার্ড) ---
def init_ia_session():
    global ia_session
    
    if not IA_LIB_AVAILABLE:
        logger.warning("⚠️ internetarchive library not installed. Only public items will be accessible.")
        return None
    
    # সরাসরি os.environ থেকে পড়ুন
    email = os.environ.get("IA_EMAIL")
    password = os.environ.get("IA_PASSWORD")
    
    if not email or not password:
        logger.warning("⚠️ IA_EMAIL or IA_PASSWORD not set. Only public items will be accessible.")
        return None
    
    try:
        logger.info(f"🔐 Logging into Internet Archive as: {email}")
        from internetarchive import configure, get_session
        configure(email, password)
        ia_session = get_session()
        # সেশন সঠিকভাবে কাজ করছে কিনা টেস্ট করুন
        test = ia_session.get('https://archive.org/account/login.php')
        if test.status_code == 200:
            logger.info("✅ Internet Archive login successful")
        else:
            logger.warning("⚠️ Login may have issues, but continuing...")
        return ia_session
    except Exception as e:
        logger.error(f"❌ Internet Archive login failed: {e}")
        return None
def init_pinecone():
    global pc, index, embedding_model

    logger.info("🔌 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        logger.info(f"✅ Created new Pinecone index: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Pinecone ready")

# --- HF Tracking ---
def load_upload_history():
    """HF থেকে আপলোড ইতিহাস লোড"""
    try:
        if hf_api:
            file_path = hf_hub_download(
                repo_id=HF_TRACKING_REPO,
                filename="uploaded_files.json",
                repo_type="dataset",
                token=HF_TOKEN
            )
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"HF থেকে লোড করা যায়নি: {e}")

    return {"files": {}, "total_uploads": 0}

def save_upload_history(history):
    """HF-এ আপলোড ইতিহাস সংরক্ষণ"""
    try:
        if hf_api:
            temp_path = "/tmp/uploaded_files.json"
            with open(temp_path, 'w') as f:
                json.dump(history, f, indent=2)

            hf_api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="uploaded_files.json",
                repo_id=HF_TRACKING_REPO,
                repo_type="dataset",
                commit_message=f"Update - {datetime.now().isoformat()}"
            )
            logger.info("✅ Tracking uploaded to HF")
            return True
    except Exception as e:
        logger.error(f"HF upload failed: {e}")
    return False

def check_file_already_uploaded(file_hash):
    """ফাইল আগে HF-এ আপলোড হয়েছে কিনা"""
    history = load_upload_history()
    return file_hash in history.get('files', {})

def mark_file_as_uploaded(filename, file_hash, pages, vectors, size_mb, volume_info=None):
    """ফাইল HF-এ আপলোড হিসেবে মার্ক"""
    history = load_upload_history()
    history['files'][file_hash] = {
        'filename': filename,
        'hash': file_hash,
        'uploaded_at': datetime.now().isoformat(),
        'pages': pages,
        'vectors': vectors,
        'size_mb': size_mb,
        'volume_info': volume_info or {}
    }
    history['total_uploads'] = len(history['files'])
    save_upload_history(history)

# --- Hash Functions ---
def get_file_hash(pdf_bytes):
    """PDF-এর ইউনিক হ্যাশ"""
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
    file_size = len(pdf_bytes)

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        return f"{content_hash[:8]}_{file_size}_{page_count}"
    except:
        return f"{content_hash[:8]}_{file_size}"

# --- Internet Archive Functions (হায়ারার্কি সহ - লগইন সেশন ব্যবহার) ---
async def get_account_folders(account_id):
    """
    অ্যাকাউন্ট → ফোল্ডার (আইটেম) লিস্ট
    লগইন সেশন থাকলে প্রাইভেট আইটেমও দেখাবে
    """
    all_items = []
    
    # যদি লগইন সেশন থাকে, তাহলে তা দিয়ে সার্চ করব
    if ia_session:
        try:
            logger.info(f"🔍 Searching with login for uploader:{account_id}")
            
            # ✅ ভিন্ন পদ্ধতি: সরাসরি সেশন দিয়ে সার্চ
            def search_items():
                # লগইন সেশন ব্যবহার করে সার্চ
                search_result = ia_session.search_items(f'uploader:{account_id}')
                return list(search_result)
            
            items = await asyncio.to_thread(search_items)
            
            for item in items:
                all_items.append({
                    'identifier': item['identifier'],
                    'title': item.get('title', item['identifier']),
                    'date': item.get('date', '')
                })
            
            logger.info(f"📁 Found {len(all_items)} items (including private) using login")
            
            # ✅ যদি কিছু না পাওয়া যায়, তাহলে আইটেম আইডি সরাসরি চেষ্টা করুন
            if len(all_items) == 0:
                logger.info("🔄 No items found via search, trying direct item access...")
                # আপনার জানা আইটেম আইডি চেষ্টা করুন
                test_item = "4_20260415_20260415_1019"
                try:
                    item_metadata = await get_item_metadata(test_item)
                    if item_metadata:
                        all_items.append({
                            'identifier': test_item,
                            'title': item_metadata.get('title', test_item),
                            'date': item_metadata.get('date', '')
                        })
                        logger.info(f"✅ Found item via direct access: {test_item}")
                except Exception as e:
                    logger.warning(f"Direct access failed: {e}")
            
            return all_items
            
        except Exception as e:
            logger.warning(f"Login-based search failed: {e}, falling back to public API")
    
    # ফ্যালব্যাক: পাবলিক API (শুধু পাবলিক আইটেম)
    logger.info("📁 Using public API (only public items)")
    # ... বাকি কোড আগের মতো থাকবে
async def get_item_metadata(item_id):
    """একটি নির্দিষ্ট আইটেমের মেটাডেটা আনে"""
    metadata_url = f"https://archive.org/metadata/{item_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(metadata_url)
        if response.status_code == 200:
            data = response.json()
            return data.get('metadata', {})
    return None
async def get_pdfs_from_folder(folder_identifier):
    """
    ফোল্ডার → আইটেম → PDF
    একটি ফোল্ডার (আইটেম) থেকে সব PDF ফাইল বের করা
    """
    metadata_url = f"https://archive.org/metadata/{folder_identifier}"

    async with httpx.AsyncClient() as client:
        response = await client.get(metadata_url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch metadata for {folder_identifier}")
            return []

        data = response.json()

        folder_title = data.get('metadata', {}).get('title', folder_identifier)
        files = data.get('files', [])

        pdf_files = []
        for f in files:
            if f.get('name', '').endswith('.pdf'):
                pdf_files.append({
                    'filename': f['name'],
                    'url': f"https://archive.org/download/{folder_identifier}/{f['name']}",
                    'size': f.get('size', 0),
                    'folder_identifier': folder_identifier,
                    'folder_title': folder_title
                })

        return pdf_files

async def download_pdf(url):
    """PDF ডাউনলোড"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")
        return response.content

# --- PDF Processing (পৃষ্ঠা-লেভেল চেকপয়েন্ট সহ) ---
def extract_text_from_page(doc, page_num):
    """একটি পৃষ্ঠা থেকে টেক্সট এক্সট্র্যাক্ট"""
    page = doc.load_page(page_num - 1)
    text = page.get_text("text")

    if (not text or not text.strip()) and OCR_AVAILABLE:
        pix = page.get_pixmap(dpi=OCR_DPI)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        try:
            text = pytesseract.image_to_string(image, lang=OCR_LANG_PRIMARY)
        except:
            text = ""
        finally:
            image.close()
            del pix
            del image

    del page
    return text

def create_chunks(text, filename, page_num, book_name, volume_number=1):
    """টেক্সট থেকে চাঙ্ক তৈরি"""
    chunks = []

    if not text or not text.strip():
        return chunks

    paras = re.split(r'\n\s*\n', text)

    for j, para in enumerate(paras):
        para = para.strip()
        if len(para) < 50:
            continue

        chunks.append({
            'text': para,
            'metadata': {
                'type': 'paragraph',
                'page': page_num,
                'para_number': j + 1,
                'filename': filename,
                'book': book_name,
                'volume': volume_number
            }
        })

    return chunks

def save_to_pinecone(chunks, filename, page_num):
    """Pinecone-এ চাঙ্ক সংরক্ষণ"""
    global index, embedding_model

    if not chunks:
        return 0

    texts = [c['text'] for c in chunks]
    embeddings = embedding_model.encode(texts, batch_size=16, show_progress_bar=False)
    embeddings = embeddings.tolist()

    vectors = []
    file_hash = hashlib.sha256(filename.encode()).hexdigest()[:8]

    for j, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        unique_id = f"{file_hash}_{page_num}_{j}_{uuid.uuid4().hex[:4]}"

        vectors.append({
            'id': unique_id,
            'values': emb,
            'metadata': {
                'text': chunk['text'][:200],
                **chunk['metadata']
            }
        })

    index.upsert(vectors=vectors)
    return len(vectors)

async def process_single_pdf(pdf_info, folder_name):
    """একটি PDF প্রক্রিয়াকরণ"""
    logger.info(f"📄 Processing: {pdf_info['filename']} (from {folder_name})")

    # ডাউনলোড
    pdf_bytes = await download_pdf(pdf_info['url'])
    file_hash = get_file_hash(pdf_bytes)

    # HF চেক
    if check_file_already_uploaded(file_hash):
        logger.info(f"⏭️ Already fully uploaded: {pdf_info['filename']}")
        return {'status': 'skipped', 'reason': 'already_uploaded'}

    # চেকপয়েন্ট চেক
    if is_file_completed(file_hash):
        logger.info(f"⏭️ Already completed (checkpoint): {pdf_info['filename']}")
        return {'status': 'skipped', 'reason': 'checkpoint_completed'}

    progress = get_file_progress(file_hash)
    start_page = progress.get('last_page', 0) + 1

    # PDF খোলা
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    if start_page > total_pages:
        logger.warning(f"Checkpoint page {start_page} exceeds total {total_pages}, restarting from 1")
        start_page = 1

    if start_page > 1:
        logger.info(f"📌 Resuming from page {start_page}/{total_pages}")

    update_file_progress(file_hash, start_page - 1, total_pages)

    total_vectors = 0

    # প্রতি পৃষ্ঠা প্রক্রিয়াকরণ
    for page_num in range(start_page, total_pages + 1):
        text = extract_text_from_page(doc, page_num)

        volume_number = 1
        match = re.search(r'[Vv]ol(?:ume)?\s*(\d+)', pdf_info['filename'])
        if match:
            volume_number = int(match.group(1))

        chunks = create_chunks(text, pdf_info['filename'], page_num, folder_name, volume_number)

        if chunks:
            uploaded = save_to_pinecone(chunks, pdf_info['filename'], page_num)
            total_vectors += uploaded

        if page_num % 5 == 0 or page_num == total_pages:
            update_file_progress(file_hash, page_num, total_pages)
            logger.info(f"   💾 Checkpoint: page {page_num}/{total_pages}")

        if page_num % 10 == 0:
            logger.info(f"   📄 Page {page_num}/{total_pages}")
            gc.collect()

    doc.close()

    update_file_progress(file_hash, total_pages, total_pages, completed=True)

    size_mb = len(pdf_bytes) / (1024 * 1024)
    volume_info = {
        'folder_name': folder_name,
        'filename': pdf_info['filename']
    }
    mark_file_as_uploaded(pdf_info['filename'], file_hash, total_pages, total_vectors, size_mb, volume_info)

    logger.info(f"✅ Completed: {pdf_info['filename']} ({total_pages} pages, {total_vectors} vectors)")

    return {
        'status': 'success',
        'pages': total_pages,
        'vectors': total_vectors,
        'size_mb': size_mb,
        'resumed_from': start_page
    }

# --- মেইন ফাংশন (হায়ারার্কি সহ) ---
async def main():
    """মেইন প্রক্রিয়া: অ্যাকাউন্ট → ফোল্ডার → PDF"""
    logger.info("=" * 60)
    logger.info("🚀 Internet Archive PDF Processor - GitHub Actions Cron")
    logger.info(f"📁 Checkpoint dir: {CHECKPOINT_DIR}")
    logger.info("=" * 60)

    # ✅ Internet Archive লগইন (ইমেইল/পাসওয়ার্ড থাকলে)
    init_ia_session()

    # HF Login
    if HF_TOKEN:
        login(token=HF_TOKEN)
        global hf_api
        hf_api = HfApi(token=HF_TOKEN)
        logger.info("✅ Hugging Face logged in")

    # Pinecone Init
    init_pinecone()

    # ✅ ধাপ ১: অ্যাকাউন্ট থেকে সব ফোল্ডার (আইটেম) বের করা
    logger.info(f"🔍 Scanning IA account: {IA_ACCOUNT_ID}")
    folders = await get_account_folders(IA_ACCOUNT_ID)
    logger.info(f"📁 Found {len(folders)} folders/items")

    if not folders:
        logger.error("❌ No folders found! Check your IA_ACCOUNT_ID and credentials.")
        logger.info("💡 Make sure your IA_ACCOUNT_ID is correct and items are either public or you have set IA_EMAIL/IA_PASSWORD for private items.")
        return

    # ✅ ধাপ ২: প্রতিটি ফোল্ডার থেকে PDF বের করা
    all_pdfs = []

    for folder in folders:
        folder_id = folder['identifier']
        folder_title = folder['title']

        logger.info(f"📂 Processing folder: {folder_title} ({folder_id})")

        pdfs = await get_pdfs_from_folder(folder_id)

        for pdf in pdfs:
            pdf['folder_title'] = folder_title
            all_pdfs.append(pdf)

        logger.info(f"   📄 Found {len(pdfs)} PDFs in this folder")

    logger.info(f"📊 Total PDFs found: {len(all_pdfs)}")

    # ✅ চেকপয়েন্ট ও HF হিস্টরি
    checkpoint = load_page_checkpoint()
    logger.info(f"💾 Checkpoint has {len(checkpoint)} entries")

    history = load_upload_history()
    logger.info(f"📋 HF history has {len(history.get('files', {}))} entries")

    # ✅ নতুন PDF ফিল্টার
    pending_pdfs = []

    for pdf in all_pdfs:
        temp_hash = hashlib.md5(f"{pdf['filename']}_{pdf['size']}".encode()).hexdigest()

        if temp_hash in history.get('files', {}):
            logger.info(f"⏭️ Skipping (HF): {pdf['filename']}")
            continue

        progress = get_file_progress(temp_hash)
        if progress.get('completed', False):
            logger.info(f"⏭️ Skipping (checkpoint): {pdf['filename']}")
            continue

        pending_pdfs.append(pdf)

    logger.info(f"🆕 Pending PDFs: {len(pending_pdfs)}")

    if not pending_pdfs:
        logger.info("✅ No pending PDFs found. Exiting.")
        return

    # ✅ একে একে প্রক্রিয়াকরণ
    processed = 0
    failed = 0
    skipped = 0
    resumed = 0

    for pdf in pending_pdfs:
        try:
            result = await process_single_pdf(pdf, pdf['folder_title'])

            if result['status'] == 'success':
                processed += 1
                if result.get('resumed_from', 1) > 1:
                    resumed += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"❌ Failed: {pdf['filename']} - {e}")
            failed += 1

        await asyncio.sleep(1)

    # ফাইনাল রিপোর্ট
    logger.info("=" * 60)
    logger.info("📊 FINAL REPORT")
    logger.info("=" * 60)
    logger.info(f"✅ Processed: {processed} ({resumed} resumed from checkpoint)")
    logger.info(f"⏭️ Skipped: {skipped}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"💾 Checkpoint entries: {len(load_page_checkpoint())}")
    logger.info(f"📁 HF tracking entries: {len(load_upload_history().get('files', {}))}")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
