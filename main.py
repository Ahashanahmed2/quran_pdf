import os
import asyncio
import io
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from pypdf import PdfReader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import logging

# --- ১. লগিং সেটআপ ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ২. কনফিগারেশন ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8613624366:AAHWX_Y_7bH5V8Mw4hfUQ0nfPaGrfZ-ROgw")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_7XHfjD_Ekff9WkF5MPke5mUwFTQ24ctf45NnvbWDXXQEozdEf8aHHHNRgH4PzpfHDwRZqE")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quranqpf")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://quran-pdf-2.onrender.com")
SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET", "asdFGH")

# --- ৩. Pinecone ও এম্বেডিং মডেল সেটআপ ---
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

# --- ৪. উন্নত PDF প্রসেসিং ফাংশন ---

def detect_headlines(page_text):
    """পৃষ্ঠা থেকে সম্ভাব্য হেডলাইন ডিটেক্ট করা"""
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
    """পৃষ্ঠা থেকে প্যারাগ্রাফ আলাদা করা"""
    raw_paras = re.split(r'\n\s*\n', page_text)
    
    paragraphs = []
    for para in raw_paras:
        para = para.strip()
        if len(para) > 25:
            para = re.sub(r'\s+', ' ', para)
            paragraphs.append(para)
    
    return paragraphs

def extract_text_from_pdf_bytes_advanced(pdf_bytes):
    """পিডিএফ থেকে টেক্সট, পৃষ্ঠা নম্বর, হেডলাইন ও প্যারাগ্রাফ সহ বের করা"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        structured_pages = []
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
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
            
            logger.info(f"   📄 Page {page_num}: {len(headlines)} headlines, {len(paragraphs)} paragraphs")
        
        return structured_pages
    except Exception as e:
        logger.error(f"PDF এক্সট্র্যাক্ট ত্রুটি: {e}")
        raise

def create_structured_chunks(structured_pages, filename):
    """স্ট্রাকচার্ড ডেটা থেকে চাঙ্ক তৈরি"""
    chunks = []
    
    for page in structured_pages:
        page_num = page['page_number']
        headlines = page['headlines']
        paragraphs = page['paragraphs']
        
        # হেডলাইনগুলো আলাদা চাঙ্ক হিসেবে
        for headline in headlines:
            chunks.append({
                'text': headline,
                'metadata': {
                    'page': page_num,
                    'type': 'headline',
                    'headline': headline[:100]
                }
            })
        
        # প্রতিটি প্যারাগ্রাফ আলাদা চাঙ্ক হিসেবে
        for para_num, para_text in enumerate(paragraphs, 1):
            related_headline = headlines[0] if headlines else "No headline"
            
            chunks.append({
                'text': para_text,
                'metadata': {
                    'page': page_num,
                    'type': 'paragraph',
                    'para_number': para_num,
                    'headline': related_headline[:100],
                    'total_paras': len(paragraphs)
                }
            })
        
        # পুরো পৃষ্ঠার টেক্সটও একটি চাঙ্ক হিসেবে
        if page['full_text'].strip():
            chunks.append({
                'text': page['full_text'][:2000],
                'metadata': {
                    'page': page_num,
                    'type': 'full_page',
                    'headline': headlines[0][:100] if headlines else "No headline"
                }
            })
    
    return chunks

def save_structured_to_pinecone(filename, chunks):
    """স্ট্রাকচার্ড চাঙ্কগুলো Pinecone-এ সংরক্ষণ"""
    if index is None or embedding_model is None:
        raise Exception("Pinecone বা এম্বেডিং মডেল লোড হয়নি")
    
    vectors = []
    for i, chunk_data in enumerate(chunks):
        chunk_text = chunk_data['text']
        metadata = chunk_data['metadata']
        
        embedding = embedding_model.encode(chunk_text).tolist()
        
        full_metadata = {
            "filename": filename,
            "chunk_index": i,
            "text": chunk_text[:1000],
            **metadata
        }
        
        vectors.append({
            "id": f"{filename.replace('.', '_')}_chunk_{i}",
            "values": embedding,
            "metadata": full_metadata
        })
    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    
    return len(vectors)

def search_in_pinecone_advanced(query, top_k=5):
    """উন্নত সার্চ - রেজাল্টে পৃষ্ঠা, হেডলাইন, প্যারা নম্বর সহ"""
    if index is None or embedding_model is None:
        return []
    
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
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
    """সার্চ রেজাল্ট সুন্দরভাবে ফরম্যাট করা"""
    if not results:
        return "❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"
    
    formatted = f"🔍 **প্রশ্ন:** {query}\n\n"
    formatted += f"📊 **প্রাপ্ত ফলাফল:** {len(results)}টি\n\n"
    
    # টাইপ অনুযায়ী গ্রুপ করা
    headlines = [r for r in results if r.get('type') == 'headline']
    paragraphs = [r for r in results if r.get('type') == 'paragraph']
    full_pages = [r for r in results if r.get('type') == 'full_page']
    
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
    
    if full_pages and not paragraphs:
        formatted += "📄 **পৃষ্ঠার তথ্য:**\n"
        for fp in full_pages[:1]:
            formatted += f"\n📍 **পৃষ্ঠা {fp['page']}**\n{fp['text'][:400]}...\n"
    
    formatted += f"\n---\n📚 **সোর্স:** {results[0].get('filename', 'Unknown')}"
    
    return formatted

# --- ৫. টেলিগ্রাম বট হ্যান্ডলার ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Quran PDF Bot**\n\n"
        "/help - সকল কমান্ড দেখুন\n"
        "/status - সিস্টেম স্ট্যাটাস\n\n"
        "✨ **ফিচার:** পৃষ্ঠা নম্বর, হেডলাইন, প্যারা নম্বর সহ তথ্য!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📚 **উপলব্ধ কমান্ডসমূহ:**\n\n"
        "/start - বট চালু করুন\n"
        "/help - এই সাহায্য বার্তা\n"
        "/file - PDF আপলোড করুন\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n"
        "/status - সিস্টেম স্ট্যাটাস\n\n"
        "**PDF আপলোড:** `/file` লিখে PDF পাঠান\n"
        "**প্রশ্ন:** সরাসরি প্রশ্ন লিখুন\n\n"
        "✨ উত্তর পৃষ্ঠা নম্বর, হেডলাইন ও প্যারা রেফারেন্সসহ আসবে!"
    )
    await update.message.reply_text(help_text)

async def handle_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("📂 handle_file_command called")
    
    if update.message.document:
        document = update.message.document
        logger.info(f"📄 Document: {document.file_name}, size: {document.file_size} bytes")
        
        if not document.file_name.endswith('.pdf'):
            await update.message.reply_text("❌ শুধুমাত্র PDF ফাইল সমর্থিত।")
            return
        
        status_msg = await update.message.reply_text(f"⏳ '{document.file_name}' ডাউনলোড হচ্ছে...")
        
        try:
            file = await context.bot.get_file(document.file_id)
            pdf_bytes = await file.download_as_bytearray()
            logger.info(f"📥 Downloaded {len(pdf_bytes)} bytes")
            
            await status_msg.edit_text("⏳ টেক্সট এক্সট্র্যাক্ট হচ্ছে (পৃষ্ঠা, হেডলাইন, প্যারা)...")
            
            structured_pages = extract_text_from_pdf_bytes_advanced(pdf_bytes)
            
            if not structured_pages:
                await status_msg.edit_text("❌ PDF থেকে কোনো টেক্সট পাওয়া যায়নি।")
                return
            
            total_pages = len(structured_pages)
            total_headlines = sum(len(p['headlines']) for p in structured_pages)
            total_paras = sum(len(p['paragraphs']) for p in structured_pages)
            
            await status_msg.edit_text(
                f"📊 প্রাপ্ত তথ্য:\n"
                f"📄 পৃষ্ঠা: {total_pages}\n"
                f"📌 হেডলাইন: {total_headlines}\n"
                f"📝 প্যারাগ্রাফ: {total_paras}\n\n"
                f"⏳ Pinecone-এ সংরক্ষণ হচ্ছে..."
            )
            
            chunks = create_structured_chunks(structured_pages, document.file_name)
            vector_count = save_structured_to_pinecone(document.file_name, chunks)
            logger.info(f"💾 Saved {vector_count} vectors to Pinecone")
            
            await status_msg.edit_text(
                f"✅ **'{document.file_name}'** সফলভাবে সংরক্ষিত!\n\n"
                f"📄 পৃষ্ঠা: {total_pages}\n"
                f"📌 হেডলাইন: {total_headlines}\n"
                f"📝 প্যারাগ্রাফ: {total_paras}\n"
                f"🗄️ মোট ভেক্টর: {vector_count}\n\n"
                f"ℹ️ এখন প্রশ্ন করলে পৃষ্ঠা, হেডলাইন ও প্যারা রেফারেন্স পাবেন!"
            )
            
        except Exception as e:
            logger.error(f"❌ PDF প্রসেসিং ত্রুটি: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ ত্রুটি: {str(e)}")
    else:
        await update.message.reply_text("📎 দয়া করে একটি PDF ফাইল পাঠান।")

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    logger.info(f"❓ Question: {user_question[:50]}...")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        results = search_in_pinecone_advanced(user_question, top_k=5)
        logger.info(f"🔍 Found {len(results)} results")
        
        formatted_answer = format_search_results(results, user_question)
        await update.message.reply_text(formatted_answer, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"প্রশ্ন প্রসেসিং ত্রুটি: {e}")
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def list_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if index is None:
            await update.message.reply_text("❌ Pinecone সংযুক্ত নয়")
            return
        
        results = index.query(
            vector=[0.1]*384,
            top_k=1000,
            include_metadata=True
        )
        
        file_stats = {}
        for match in results.matches:
            if match.metadata and 'filename' in match.metadata:
                filename = match.metadata['filename']
                if filename not in file_stats:
                    file_stats[filename] = {'pages': set(), 'headlines': 0, 'paras': 0}
                
                page = match.metadata.get('page')
                if page:
                    file_stats[filename]['pages'].add(page)
                if match.metadata.get('headline') and match.metadata['headline'] != 'N/A':
                    file_stats[filename]['headlines'] += 1
                if match.metadata.get('type') == 'paragraph':
                    file_stats[filename]['paras'] += 1
        
        if file_stats:
            file_list = ""
            for filename, stats in file_stats.items():
                file_list += f"\n📁 **{filename}**\n"
                file_list += f"   📄 পৃষ্ঠা: {len(stats['pages'])}\n"
                file_list += f"   📌 হেডলাইন: {stats['headlines']}\n"
                file_list += f"   📝 প্যারাগ্রাফ: {stats['paras']}\n"
            
            await update.message.reply_text(f"**সংরক্ষিত PDF:**\n{file_list}", parse_mode="Markdown")
        else:
            await update.message.reply_text("ℹ️ এখনো কোনো PDF সংরক্ষিত হয়নি।")
            
    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pc_status = "✅" if index is not None else "❌"
    status_text = (
        f"📊 **সিস্টেম স্ট্যাটাস**\n\n"
        f"🗄️ Pinecone: {pc_status}\n"
        f"📚 ইনডেক্স: {PINECONE_INDEX_NAME}\n\n"
        f"✨ **ফিচার:** পৃষ্ঠা, হেডলাইন, প্যারা নম্বর সহ তথ্য!"
    )
    await update.message.reply_text(status_text, parse_mode="Markdown")

# --- ৬. FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    request = HTTPXRequest(connection_pool_size=10, read_timeout=120, write_timeout=120)
    bot = Bot(token=TELEGRAM_TOKEN, request=request)
    
    ptb_app = Application.builder().bot(bot).build()
    
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("help", help_command))
    ptb_app.add_handler(CommandHandler("file", handle_file_command))
    ptb_app.add_handler(CommandHandler("list", list_files))
    ptb_app.add_handler(CommandHandler("status", status))
    ptb_app.add_handler(MessageHandler(filters.Document.PDF, handle_file_command))
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

# --- ৭. রুট এন্ডপয়েন্ট ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Bot is running", "service": "Quran PDF Bot"}

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
    print(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")