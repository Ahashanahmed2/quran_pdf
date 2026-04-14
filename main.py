import os
import asyncio
import io
import re
import json
import hashlib
import requests
import gc
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
    logger_ocr = logging.getLogger(__name__)
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available. Install: pytesseract, pdf2image")

# --- ১. লগিং সেটআপ ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ২. কনফিগারেশন ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8613624366:AAHWX_Y_7bH5V8Mw4hfUQ0nfPaGrfZ-ROgw")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_7XHfjD_Ekff9WkF5MPke5mUwFTQ24ctf45NnvbWDXXQEozdEf8aHHHNRgH4PzpfHDwRZqE")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quranqpf")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://quran-pdf-2.onrender.com")
SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET", "asdFGH")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IA_EMAIL = os.environ.get("IA_EMAIL", "")
IA_PASSWORD = os.environ.get("IA_PASSWORD", "")
GOOGLE_DRIVE_API_KEY = os.environ.get("GOOGLE_DRIVE_API_KEY", "")

HF_TRACKING_REPO = "ahashanahmed/quran-bot-tracking"

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

# --- ৬. HF ট্র্যাকিং ফাংশন ---

def load_upload_history():
    try:
        if hf_api:
            file_path = hf_hub_download(
                repo_id=HF_TRACKING_REPO,
                filename="uploaded_files.json",
                repo_type="dataset",
                token=HF_TOKEN
            )
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"HF থেকে লোড করা যায়নি: {e}")
    return {"files": {}, "total_uploads": 0}

def save_upload_history(history):
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

# --- ৭. Archive.org ফাংশন ---

def upload_to_archive(pdf_bytes, filename, title=""):
    try:
        file_hash = hashlib.md5(pdf_bytes).hexdigest()[:10]
        
        safe_filename = re.sub(r'[^\x20-\x7E]', '_', filename)
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', safe_filename)
        safe_filename = re.sub(r'_+', '_', safe_filename)
        safe_filename = safe_filename.strip('_')
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

def get_file_list_from_folder(folder_id):
    files = []
    if GOOGLE_DRIVE_API_KEY:
        url = "https://www.googleapis.com/drive/v3/files"
        params = {
            "q": f"'{folder_id}' in parents and mimeType='application/pdf'",
            "fields": "files(id, name, size)",
            "key": GOOGLE_DRIVE_API_KEY,
            "pageSize": 100
        }
        page_token = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                files.extend(data.get('files', []))
                page_token = data.get('nextPageToken')
                if not page_token:
                    break
            else:
                logger.error(f"Google Drive API error: {response.status_code}")
                break
    return files

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
    """টেক্সট-বেসড PDF থেকে টেক্সট এক্সট্র্যাক্ট"""
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
    """OCR ব্যবহার করে স্ক্যান করা PDF থেকে টেক্সট এক্সট্র্যাক্ট"""
    if not OCR_AVAILABLE:
        logger.error("❌ OCR libraries not available")
        return []
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)  # DPI কমানো হয়েছে মেমোরির জন্য
        total_pages = len(images)
        structured_pages = []
        
        logger.info(f"📚 Total pages: {total_pages}, processing with OCR in batches of {batch_size}")
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            logger.info(f"🖼️ Processing OCR batch: pages {batch_start+1}-{batch_end} of {total_pages}")
            
            for page_num in range(batch_start, batch_end):
                image = images[page_num]
                
                # OCR দিয়ে টেক্সট এক্সট্র্যাক্ট (বাংলা + ইংরেজি)
                try:
                    page_text = pytesseract.image_to_string(image, lang='ben+eng')
                except:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                
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
                
                if (page_num + 1) % 5 == 0:
                    logger.info(f"   📄 Page {page_num + 1}: {len(headlines)} headlines, {len(paragraphs)} paragraphs")
            
            gc.collect()
            logger.info(f"   🧹 Memory cleared after OCR batch")
        
        logger.info(f"✅ OCR extraction complete: {len(structured_pages)} pages processed")
        return structured_pages
        
    except Exception as e:
        logger.error(f"OCR এক্সট্র্যাক্ট ত্রুটি: {e}")
        raise

def extract_text_from_pdf_bytes_auto(pdf_bytes):
    """অটো-ডিটেক্ট: PDF টেক্সট-বেসড না স্ক্যান করা"""
    try:
        # প্রথমে টেক্সট-বেসড হিসেবে ট্রাই
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) == 0:
            return []
        
        first_page_text = reader.pages[0].extract_text() if len(reader.pages) > 0 else ""
        
        # যদি প্রথম পৃষ্ঠায় ৫০-এর কম অক্ষর থাকে, তাহলে স্ক্যান করা PDF ধরে OCR ব্যবহার
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
    
    for i, chunk_data in enumerate(chunks):
        chunk_text = chunk_data['text']
        metadata = chunk_data['metadata']
        embedding = embedding_model.encode(chunk_text).tolist()
        
        full_metadata = {
            "filename": filename, "chunk_index": i, "text": chunk_text[:1000], **metadata
        }
        
        vectors.append({
            "id": f"{filename.replace('.', '_')}_chunk_{i}",
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
    
    formatted = f"🔍 **প্রশ্ন:** {query}\n\n📊 **প্রাপ্ত ফলাফল:** {len(results)}টি\n\n"
    
    headlines = [r for r in results if r.get('type') == 'headline']
    paragraphs = [r for r in results if r.get('type') == 'paragraph']
    
    if headlines:
        formatted += "📌 **প্রাসঙ্গিক হেডলাইন:**\n"
        for h in headlines[:3]:
            formatted += f"• [পৃষ্ঠা {h['page']}] {h['text'][:100]}...\n"
        formatted += "\n"
    
    if paragraphs:
        formatted += "📝 **প্রাসঙ্গিক অংশ:**\n"
        for p in paragraphs[:2]:
            formatted += f"\n📍 **পৃষ্ঠা {p['page']}**"
            if p.get('headline') and p['headline'] != 'N/A':
                formatted += f" | {p['headline'][:40]}"
            if p.get('para_number') and p['para_number'] != 'N/A':
                formatted += f" | প্যারা {p['para_number']}"
            formatted += f"\n{p['text'][:300]}...\n"
    
    formatted += f"\n---\n📚 **সোর্স:** {results[0].get('filename', 'Unknown')}"
    return formatted

# --- ১০. Telegram বট হ্যান্ডলার ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Quran PDF Bot**\n\n"
        "/help - সকল কমান্ড দেখুন\n"
        "/status - সিস্টেম স্ট্যাটাস\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n\n"
        "✨ Google Drive লিংক দিন বড় PDF আপলোড করতে!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📚 **উপলব্ধ কমান্ডসমূহ:**\n\n"
        "/start - বট চালু করুন\n"
        "/help - এই সাহায্য বার্তা\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n"
        "/status - সিস্টেম স্ট্যাটাস\n\n"
        "**Google Drive থেকে PDF আপলোড:**\n"
        "1. PDF Google Drive-এ আপলোড করুন\n"
        "2. শেয়ারেবল লিংক কপি করুন\n"
        "3. লিংকটি এখানে পেস্ট করুন\n\n"
        "**প্রশ্ন:** সরাসরি প্রশ্ন লিখুন"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def handle_drive_folder(update: Update, context: ContextTypes.DEFAULT_TYPE, folder_id):
    status_msg = await update.message.reply_text("📁 ফোল্ডার স্ক্যান করা হচ্ছে...")
    files = get_file_list_from_folder(folder_id)
    
    if not files:
        await status_msg.edit_text("ℹ️ এই ফোল্ডারে কোনো PDF ফাইল পাওয়া যায়নি।")
        return
    
    new_files = []
    duplicate_files = []
    
    for file in files:
        temp_hash = hashlib.md5(f"{file['name']}_{file.get('size', '')}".encode()).hexdigest()
        if check_file_already_uploaded(temp_hash)['exists']:
            duplicate_files.append(file)
        else:
            new_files.append(file)
    
    report = f"📊 **ফোল্ডার স্ক্যান রিপোর্ট**\n\n"
    report += f"📁 মোট PDF: {len(files)}টি\n"
    report += f"🆕 নতুন: {len(new_files)}টি\n"
    report += f"⏭️ আগে আপলোডকৃত: {len(duplicate_files)}টি\n\n"
    
    if duplicate_files:
        report += "**আগে আপলোডকৃত (স্কিপ হবে):**\n"
        for f in duplicate_files[:5]:
            report += f"• {f['name']}\n"
        if len(duplicate_files) > 5:
            report += f"...এবং আরও {len(duplicate_files) - 5}টি\n"
        report += "\n"
    
    if not new_files:
        report += "ℹ️ প্রক্রিয়াকরণের জন্য কোনো নতুন ফাইল নেই।"
        await status_msg.edit_text(report, parse_mode="Markdown")
        return
    
    report += f"⏳ {len(new_files)}টি নতুন ফাইল প্রক্রিয়াকরণ শুরু..."
    await status_msg.edit_text(report, parse_mode="Markdown")
    
    processed = 0
    failed = 0
    
    for file in new_files:
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file['id']}"
            response = requests.get(download_url, timeout=120)
            
            if response.status_code != 200:
                failed += 1
                continue
            
            pdf_bytes = response.content
            file_hash = get_file_hash(pdf_bytes)
            filename = file['name']
            size_mb = len(pdf_bytes) / 1024 / 1024
            
            archive_result = upload_to_archive(pdf_bytes, filename)
            archive_url = archive_result.get('url') if archive_result['success'] else None
            
            structured_pages = extract_text_from_pdf_bytes_auto(pdf_bytes)
            chunks = create_structured_chunks(structured_pages, filename)
            vector_count = save_structured_to_pinecone(filename, chunks)
            
            mark_file_as_uploaded(filename, file_hash, archive_url, len(structured_pages), vector_count, size_mb)
            
            processed += 1
            
        except Exception as e:
            logger.error(f"Failed: {file['name']} - {e}")
            failed += 1
    
    final_report = f"✅ **ফোল্ডার প্রক্রিয়াকরণ সম্পন্ন!**\n\n"
    final_report += f"✅ সফল: {processed}টি\n"
    final_report += f"❌ ব্যর্থ: {failed}টি"
    
    await update.message.reply_text(final_report, parse_mode="Markdown")

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
    
    status_msg = await update.message.reply_text("⏳ Google Drive থেকে PDF ডাউনলোড করা হচ্ছে...")
    
    try:
        download_url = f"https://drive.google.com/uc?export=download&id={folder_id}"
        response = requests.get(download_url, timeout=120)
        
        if response.status_code != 200:
            await status_msg.edit_text(f"❌ ডাউনলোড ব্যর্থ: {response.status_code}")
            return
        
        pdf_bytes = response.content
        file_hash = get_file_hash(pdf_bytes)
        
        duplicate = check_file_already_uploaded(file_hash)
        if duplicate['exists']:
            await status_msg.edit_text(
                f"⚠️ **এই ফাইলটি আগেই আপলোড করা হয়েছে!**\n\n"
                f"📅 আগের আপলোড: {duplicate['date']}\n"
                f"🔗 Archive.org: {duplicate['archive_url']}",
                parse_mode="Markdown"
            )
            return
        
        filename = f"drive_{file_hash[:8]}.pdf"
        size_mb = len(pdf_bytes) / 1024 / 1024
        
        await status_msg.edit_text(f"✅ ডাউনলোড সম্পন্ন ({size_mb:.2f} MB)৷\n⏳ Archive.org-এ আপলোড করা হচ্ছে...")
        
        archive_result = upload_to_archive(pdf_bytes, filename)
        archive_url = archive_result.get('url') if archive_result['success'] else None
        
        await status_msg.edit_text("⏳ PDF প্রক্রিয়াকরণ ও Pinecone-এ সংরক্ষণ করা হচ্ছে...")
        
        # ✅ অটো-ডিটেক্ট OCR
        structured_pages = extract_text_from_pdf_bytes_auto(pdf_bytes)
        chunks = create_structured_chunks(structured_pages, filename)
        vector_count = save_structured_to_pinecone(filename, chunks)
        
        total_pages = len(structured_pages)
        total_headlines = sum(len(p['headlines']) for p in structured_pages)
        total_paras = sum(len(p['paragraphs']) for p in structured_pages)
        
        mark_file_as_uploaded(filename, file_hash, archive_url, total_pages, vector_count, size_mb)
        
        final_msg = f"✅ **সফলভাবে সংরক্ষিত!**\n\n"
        final_msg += f"📄 পৃষ্ঠা: {total_pages}\n"
        final_msg += f"📌 হেডলাইন: {total_headlines}\n"
        final_msg += f"📝 প্যারাগ্রাফ: {total_paras}\n"
        final_msg += f"🗄️ ভেক্টর: {vector_count}\n"
        final_msg += f"📊 আকার: {size_mb:.2f} MB\n"
        
        if archive_url:
            final_msg += f"\n📚 **Archive.org:** {archive_url}"
        
        await status_msg.edit_text(final_msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"❌ Drive link error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ ত্রুটি: {str(e)}")

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    
    if 'drive.google.com' in user_question:
        await handle_drive_link(update, context)
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        results = search_in_pinecone_advanced(user_question, top_k=5)
        formatted_answer = format_search_results(results, user_question)
        await update.message.reply_text(formatted_answer, parse_mode="Markdown")
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
                file_list += f"\n📁 **{filename}**\n"
                file_list += f"   📄 পৃষ্ঠা: {len(stats['pages'])}\n"
                file_list += f"   🗄️ ভেক্টর: {stats['vectors']}\n"
            await update.message.reply_text(f"**সংরক্ষিত PDF:**\n{file_list}", parse_mode="Markdown")
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
        f"📊 **সিস্টেম স্ট্যাটাস**\n\n"
        f"🗄️ Pinecone: {pc_status}\n"
        f"📚 Archive.org: {ia_status}\n"
        f"🔍 HF Tracking: {hf_status}\n"
        f"📷 OCR: {ocr_status}\n"
        f"📁 ইনডেক্স: {PINECONE_INDEX_NAME}"
    )
    await update.message.reply_text(status_text, parse_mode="Markdown")

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