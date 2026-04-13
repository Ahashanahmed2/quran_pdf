import os
import asyncio
import io
import httpx  # ✅ requests-এর পরিবর্তে সম্পূর্ণ async httpx
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
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quran-pdf-index")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://quran-pdf-2.onrender.com")
SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET", "asdFGH")

# Hugging Face Inference API কনফিগ
HF_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-it"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

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

# --- ৪. PDF প্রসেসিং ফাংশন ---
def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        return full_text
    except Exception as e:
        logger.error(f"PDF এক্সট্র্যাক্ট ত্রুটি: {e}")
        raise

def chunk_text(text, chunk_size=500):
    if not text:
        return []
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def save_to_pinecone(filename, chunks):
    if index is None or embedding_model is None:
        raise Exception("Pinecone বা এম্বেডিং মডেল লোড হয়নি")

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vectors.append({
            "id": f"{filename.replace('.', '_')}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "filename": filename,
                "chunk_index": i,
                "text": chunk[:1000]
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

    return len(vectors)

def search_in_pinecone(query, top_k=3):
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
                chunks.append({
                    "text": match.metadata["text"],
                    "filename": match.metadata["filename"],
                    "score": match.score
                })

        return chunks
    except Exception as e:
        logger.error(f"Pinecone সার্চ ত্রুটি: {e}")
        return []

# ✅ HF API call সম্পূর্ণ async
async def generate_answer_with_gemma(question, context_chunks):
    if not context_chunks:
        return "❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি। দয়া করে আগে PDF আপলোড করুন।"

    if not HF_API_KEY:
        return "⚠️ HF_API_KEY সেট করা নেই।"

    context_text = "\n\n---\n\n".join([f"[উৎস: {c['filename']}]\n{c['text'][:500]}" for c in context_chunks])

    prompt = f"""<|turn|>system
তুমি একটি সহায়ক AI সহকারী। নিচের প্রসঙ্গ তথ্যের ভিত্তিতে ব্যবহারকারীর প্রশ্নের উত্তর দাও।
<turn|>
<|turn|>user
প্রসঙ্গ তথ্য:
{context_text}

প্রশ্ন: {question}
<turn|>
<|turn|>model
"""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 256,
                        "temperature": 0.3,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
            )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'কোনো উত্তর পাওয়া যায়নি।').strip()
            elif isinstance(result, dict):
                return result.get('generated_text', 'কোনো উত্তর পাওয়া যায়নি।').strip()
        elif response.status_code == 503:
            return "⏳ মডেল লোড হচ্ছে, কয়েক সেকেন্ড পর আবার চেষ্টা করুন।"
        else:
            logger.error(f"HF API ত্রুটি: {response.status_code}")
            return f"❌ HF API ত্রুটি: {response.status_code}"

    except Exception as e:
        logger.error(f"Gemma API ত্রুটি: {e}")
        return f"❌ ত্রুটি: {str(e)}"

# --- ৫. টেলিগ্রাম বট হ্যান্ডলার ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Quran PDF Bot\n\n"
        "/help - সকল কমান্ড দেখুন\n"
        "/status - সিস্টেম স্ট্যাটাস"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "উপলব্ধ কমান্ডসমূহ:\n\n"
        "/start - বট চালু করুন\n"
        "/help - এই সাহায্য বার্তা\n"
        "/file - PDF আপলোড করুন\n"
        "/list - সংরক্ষিত PDF-র তালিকা\n"
        "/status - সিস্টেম স্ট্যাটাস\n\n"
        "PDF আপলোড: /file লিখে PDF পাঠান\n"
        "প্রশ্ন: সরাসরি প্রশ্ন লিখুন"
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

            await status_msg.edit_text("⏳ টেক্সট এক্সট্র্যাক্ট হচ্ছে...")

            full_text = extract_text_from_pdf_bytes(pdf_bytes)
            logger.info(f"📝 Extracted {len(full_text)} characters")

            if not full_text.strip():
                await status_msg.edit_text("❌ PDF থেকে কোনো টেক্সট পাওয়া যায়নি।")
                return

            chunks = chunk_text(full_text)
            logger.info(f"🔹 Created {len(chunks)} chunks")

            if not chunks:
                await status_msg.edit_text("❌ টেক্সট খণ্ড তৈরি করা যায়নি।")
                return

            await status_msg.edit_text("⏳ Pinecone-এ সংরক্ষণ হচ্ছে...")
            vector_count = save_to_pinecone(document.file_name, chunks)
            logger.info(f"💾 Saved {vector_count} vectors to Pinecone")

            await status_msg.edit_text(
                f"✅ '{document.file_name}' সফলভাবে সংরক্ষিত!\n"
                f"📄 টেক্সট খণ্ড: {len(chunks)}\n"
                f"🗄️ ভেক্টর: {vector_count}"
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
        results = search_in_pinecone(user_question, top_k=3)
        logger.info(f"🔍 Found {len(results)} results")

        if not results:
            await update.message.reply_text("❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি। দয়া করে আগে PDF আপলোড করুন।")
            return

        # ✅ async কল
        answer = await generate_answer_with_gemma(user_question, results)
        await update.message.reply_text(answer)

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

        filenames = set()
        for match in results.matches:
            if match.metadata and 'filename' in match.metadata:
                filenames.add(match.metadata['filename'])

        if filenames:
            file_list = "\n".join([f"• {f}" for f in sorted(filenames)])
            await update.message.reply_text(f"সংরক্ষিত PDF:\n{file_list}")
        else:
            await update.message.reply_text("ℹ️ এখনো কোনো PDF সংরক্ষিত হয়নি।")

    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    hf_status = "✅" if HF_API_KEY else "❌"
    pc_status = "✅" if index is not None else "❌"
    status_text = (
        f"সিস্টেম স্ট্যাটাস\n"
        f"Pinecone: {pc_status}\n"
        f"HF API: {hf_status}"
    )
    await update.message.reply_text(status_text)

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
    """Uptime monitor-এর জন্য lightweight endpoint"""
    return {"pong": True}

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    #if request.headers.get('X-Telegram-Bot-Api-Secret-Token') != SECRET_TOKEN:
        #return Response(status_code=403)

    ptb_app = request.app.state.ptb_app
    bot = request.app.state.bot

    data = await request.json()
    update = Update.de_json(data, bot)

    # ✅ সরাসরি await - রিকোয়েস্ট alive থাকে
    await ptb_app.process_update(update)

    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
