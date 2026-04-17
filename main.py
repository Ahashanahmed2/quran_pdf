#!/usr/bin/env python3
"""
একক ফাইল: FastAPI + Telegram Bot
MediaFire থেকে PDF নিয়ে Hugging Face-এ ইমেজ আপলোড করে
"""

import os
import io
import re
import json
import time
import asyncio
import threading
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import fitz  # PyMuPDF
from mediafire import MediaFireApi
from huggingface_hub import HfApi, upload_file

# ============ কনফিগারেশন ============
app = FastAPI()

# এনভায়রনমেন্ট ভেরিয়েবল
MEDIAFIRE_EMAIL = os.environ.get("MEDIAFIRE_EMAIL")
MEDIAFIRE_PASSWORD = os.environ.get("MEDIAFIRE_PASSWORD")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# টেম্প ফোল্ডার
TEMP_DIR = Path("/tmp/tafsir_temp")
TEMP_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = Path("/tmp/checkpoint.json")
# ====================================

# ============ Pydantic মডেল ============
class ProcessRequest(BaseModel):
    folder_key: str
    folder_name: str
    telegram_chat_id: int

# ============ টেলিগ্রাম ফাংশন ============
def send_telegram(chat_id, message):
    """টেলিগ্রামে মেসেজ পাঠায়"""
    if TELEGRAM_BOT_TOKEN:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": chat_id, "text": message[:4000], "parse_mode": "Markdown"}
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            print(f"Telegram error: {e}")

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
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "current": None, "last_page": 0}

def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def get_mediafire_files(folder_key):
    """MediaFire থেকে সব PDF-এর লিংক বের করে"""
    api = MediaFireApi()
    session = api.user_get_session_token(
        email=MEDIAFIRE_EMAIL, 
        password=MEDIAFIRE_PASSWORD, 
        app_id='42511'
    )
    api.session = session
    
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

def pdf_to_images(pdf_bytes, dpi=300):
    """PDF থেকে PNG ইমেজ তৈরি করে"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)
    
    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        
        images.append({
            'page_num': page_num + 1,
            'bytes': img_bytes,
            'name': f"page_{page_num+1:04d}.png",
        })
        del pix
        
    pdf_document.close()
    return images, total_pages

def upload_to_hf(folder_path, image):
    """ইমেজ Hugging Face-এ আপলোড করে"""
    path_in_repo = f"{folder_path}/{image['name']}"
    upload_file(
        path_or_fileobj=io.BytesIO(image['bytes']),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET,
        repo_type="dataset",
        token=HF_TOKEN
    )
    return True

def download_pdf(url):
    """PDF ডাউনলোড করে"""
    response = requests.get(url, stream=True, timeout=180)
    response.raise_for_status()
    return response.content

def process_pdfs(folder_key: str, folder_name: str, chat_id: int):
    """ব্যাকগ্রাউন্ডে PDF প্রসেস করে"""
    clean_folder_name = folder_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    
    send_telegram(chat_id, f"🚀 *প্রসেসিং শুরু হয়েছে!*\n\n📁 ফোল্ডার: `{clean_folder_name}`\n🔑 কী: `{folder_key}`")
    
    try:
        pdf_files = get_mediafire_files(folder_key)
        send_telegram(chat_id, f"📚 {len(pdf_files)}টি PDF পাওয়া গেছে।")
        
        if not pdf_files:
            send_telegram(chat_id, "❌ কোনো PDF পাওয়া যায়নি!")
            return
        
        checkpoint = load_checkpoint()
        processed = set(checkpoint.get('processed', []))
        
        for pdf in pdf_files:
            if pdf['name'] in processed:
                send_telegram(chat_id, f"⏭️ স্কিপ: {pdf['name']} (ইতিমধ্যে প্রসেস হয়েছে)")
                continue
            
            sub_folder = str(pdf['number'])
            full_hf_path = f"{clean_folder_name}/{sub_folder}"
            
            send_telegram(chat_id, f"📄 *প্রসেসিং: {pdf['name']}*\n📁 লোকেশন: `{full_hf_path}`")
            
            pdf_bytes = download_pdf(pdf['download_link'])
            images, total_pages = pdf_to_images(pdf_bytes, dpi=300)
            send_telegram(chat_id, f"🖼️ {total_pages} পৃষ্ঠা কনভার্ট হয়েছে। আপলোড শুরু...")
            
            for i, img in enumerate(images):
                upload_to_hf(full_hf_path, img)
                
                if (i + 1) % 10 == 0:
                    send_telegram(chat_id, f"📊 {sub_folder}: {i+1}/{total_pages} পৃষ্ঠা আপলোড হয়েছে")
                
                time.sleep(0.3)
            
            processed.add(pdf['name'])
            checkpoint['processed'] = list(processed)
            save_checkpoint(checkpoint)
            
            send_telegram(chat_id, f"✅ *সম্পন্ন: {pdf['name']}*\n📄 {total_pages} পৃষ্ঠা, 🖼️ 300 DPI কোয়ালিটি")
        
        send_telegram(chat_id, f"🎉 *সব প্রসেস সম্পন্ন!*\n\n📁 ডেটাসেট: `{HF_DATASET}/{clean_folder_name}`")
        
    except Exception as e:
        send_telegram(chat_id, f"❌ *এরর:* `{str(e)[:200]}`")

# ============ FastAPI এন্ডপয়েন্ট ============
@app.get("/")
def root():
    return {"status": "ok", "message": "Tafsir Image Processor is running"}

@app.post("/start_processing")
async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks):
    """প্রসেসিং শুরু করার এন্ডপয়েন্ট"""
    if not request.folder_key or not request.folder_name:
        raise HTTPException(status_code=400, detail="folder_key and folder_name required")
    
    background_tasks.add_task(
        process_pdfs, 
        request.folder_key, 
        request.folder_name, 
        request.telegram_chat_id
    )
    
    return {"status": "started", "message": "Processing started in background"}

@app.get("/status")
def get_status():
    """বর্তমান অবস্থা দেখায়"""
    checkpoint = load_checkpoint()
    return {
        "processed": len(checkpoint.get('processed', [])),
        "current": checkpoint.get('current'),
        "last_page": checkpoint.get('last_page', 0)
    }

# ============ টেলিগ্রাম বট হ্যান্ডলার ============
async def tg_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 *MediaFire to HuggingFace Processor*\n\n"
        "আমাকে একটি MediaFire ফোল্ডার লিংক দিন।\n"
        "আমি নিজেই সব ফোল্ডার ও ফাইল খুঁজে বের করে\n"
        "Hugging Face-এ ইমেজ আপলোড করব।\n\n"
        "📎 উদাহরণ:\n"
        "`https://www.mediafire.com/folder/ise60co8v4h6b/তাফসীর+ফী+যিলালিল+কোরআন`\n\n"
        "লিংক পাঠান এখন!",
        parse_mode="Markdown"
    )

async def tg_handle_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.strip()
    
    folder_key, folder_name = extract_from_mediafire_url(url)
    
    if not folder_key:
        await update.message.reply_text("❌ ভুল লিংক! সঠিক MediaFire ফোল্ডার লিংক দিন।")
        return
    
    await update.message.reply_text(
        f"📁 *ফোল্ডারের তথ্য:*\n\n"
        f"🔑 কী: `{folder_key}`\n"
        f"📂 নাম: `{folder_name}`\n\n"
        f"✅ প্রসেসিং শুরু করতে /confirm",
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
    
    await update.message.reply_text(
        f"🚀 *প্রসেসিং শুরু হচ্ছে...*\n\n"
        f"📂 ফোল্ডার: `{folder_name}`\n\n"
        f"আমি অগ্রগতি জানিয়ে থাকব।",
        parse_mode="Markdown"
    )
    
    # নিজের API-তে রিকোয়েস্ট (একই সার্ভার)
    try:
        response = requests.post(
            f"http://localhost:{os.environ.get('PORT', 8000)}/start_processing",
            json={
                "folder_key": folder_key,
                "folder_name": folder_name,
                "telegram_chat_id": update.effective_chat.id
            },
            timeout=5
        )
        
        if response.status_code == 200:
            await update.message.reply_text("✅ প্রসেসিং শুরু হয়েছে!\nসমাপ্ত হলে আমি জানিয়ে দেব।")
        else:
            await update.message.reply_text(f"⚠️ সার্ভার এরর: {response.status_code}")
    except Exception as e:
        await update.message.reply_text(f"❌ সংযোগ এরর: {e}")

async def tg_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("❌ বাতিল করা হয়েছে।")

async def tg_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = requests.get(f"http://localhost:{os.environ.get('PORT', 8000)}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            await update.message.reply_text(
                f"📊 *বর্তমান অবস্থা:*\n\n"
                f"✅ প্রসেস হয়েছে: {data.get('processed', 0)}টি PDF\n"
                f"🔄 চলমান: {data.get('current', 'কিছু না')}\n"
                f"📄 শেষ পৃষ্ঠা: {data.get('last_page', 0)}",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("⚠️ সার্ভার থেকে রেসপন্স পাওয়া যায়নি।")
    except Exception as e:
        await update.message.reply_text(f"❌ অবস্থা পাওয়া যায়নি: {e}")

def run_telegram_bot():
    """টেলিগ্রাম বট আলাদা থ্রেডে চালায়"""
    tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    tg_app.add_handler(CommandHandler('start', tg_start))
    tg_app.add_handler(CommandHandler('confirm', tg_confirm))
    tg_app.add_handler(CommandHandler('cancel', tg_cancel))
    tg_app.add_handler(CommandHandler('status', tg_status))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, tg_handle_link))
    
    print("🤖 Telegram Bot is running...")
    tg_app.run_polling(allowed_updates=Update.ALL_TYPES)

# ============ মেইন ============
if __name__ == "__main__":
    import uvicorn
    
    # টেলিগ্রাম বট আলাদা থ্রেডে শুরু
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    bot_thread.start()
    
    # FastAPI সার্ভার শুরু
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 FastAPI server running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)