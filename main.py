import os
import asyncio
import io
import requests
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
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quran-pdf-index")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://your-app.onrender.com")
SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET", "my_super_secret_token_2026")

# Hugging Face Inference API কনফিগ
HF_API_URL = "https://api-inference.huggingface.co/models/google/gemma-4-E2B-it"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# --- ৩. Pinecone ও এম্বেডিং মডেল সেটআপ ---
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

# --- ৪. PDF প্রসেসিং ফাংশন ---
def extract_text_from_pdf_bytes(pdf_bytes):
    """পিডিএফ বাইট থেকে টেক্সট বের করা"""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    return full_text

def chunk_text(text, chunk_size=500):
    """টেক্সটকে ৫০০ শব্দের খণ্ডে ভাগ করা"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def save_to_pinecone(filename, chunks):
    """টেক্সট খণ্ডগুলো এম্বেড করে Pinecone-এ সংরক্ষণ"""
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vectors.append({
            "id": f"{filename}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "filename": filename,
                "chunk_index": i,
                "text": chunk
            }
        })
    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    
    return len(vectors)

def search_in_pinecone(query, top_k=3):
    """Pinecone-এ প্রশ্নের প্রাসঙ্গিক তথ্য খোঁজা"""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    chunks = []
    for match in results.matches:
        if match.score > 0.3:
            chunks.append({
                "text": match.metadata["text"],
                "filename": match.metadata["filename"],
                "score": match.score
            })
    
    return chunks

# --- ৫. Hugging Face Gemma API দিয়ে উত্তর জেনারেশন ---
def generate_answer_with_gemma(question, context_chunks):
    """Hugging Face Inference API ব্যবহার করে Gemma দিয়ে উত্তর জেনারেট করা"""
    
    if not context_chunks:
        return "❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি। দয়া করে আগে /file দিয়ে PDF আপলোড করুন।"
    
    if not HF_API_KEY:
        return "⚠️ HF_API_KEY সেট করা নেই। দয়া করে Environment Variable-এ Hugging Face API Key যোগ করুন।"
    
    # কনটেক্সট তৈরি
    context_text = "\n\n---\n\n".join([f"[উৎস: {c['filename']}]\n{c['text']}" for c in context_chunks])
    
    # Gemma 4 স্টাইলে প্রম্পট তৈরি
    prompt = f"""<|turn|>system
তুমি একটি সহায়ক AI সহকারী। নিচের প্রসঙ্গ তথ্যের ভিত্তিতে ব্যবহারকারীর প্রশ্নের উত্তর দাও। উত্তর যেন তথ্যভিত্তিক ও নির্ভুল হয়। যদি প্রসঙ্গে উত্তর না থাকে, তাহলে "প্রদত্ত তথ্যে উত্তর পাওয়া যায়নি" বলবে।
<turn|>
<|turn|>user
প্রসঙ্গ তথ্য:
{context_text}

প্রশ্ন: {question}
<turn|>
<|turn|>model
"""
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.3,
                    "do_sample": True,
                    "return_full_text": False
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'কোনো উত্তর পাওয়া যায়নি।').strip()
            elif isinstance(result, dict):
                return result.get('generated_text', 'কোনো উত্তর পাওয়া যায়নি।').strip()
        elif response.status_code == 503:
            logger.info("মডেল লোড হচ্ছে, ১০ সেকেন্ড অপেক্ষা করে আবার চেষ্টা করুন...")
            return "⏳ মডেল লোড হচ্ছে, দয়া করে কয়েক সেকেন্ড পর আবার চেষ্টা করুন।"
        else:
            logger.error(f"HF API ত্রুটি: {response.status_code} - {response.text}")
            return f"❌ HF API ত্রুটি: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏳ অনুরোধ সময় শেষ হয়েছে। দয়া করে আবার চেষ্টা করুন।"
    except Exception as e:
        logger.error(f"Gemma API ত্রুটি: {e}")
        return f"❌ ত্রুটি: {str(e)}"

# --- ৬. টেলিগ্রাম বট হ্যান্ডলার ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Quran PDF Bot** (Gemma 4 E2B + Hugging Face)\n\n"
        "**ব্যবহার পদ্ধতি:**\n"
        "• `/file` কমান্ডের সাথে PDF আপলোড করুন\n"
        "• সরাসরি প্রশ্ন লিখলে Gemma PDF থেকে উত্তর দেবে\n"
        "• `/list` দিয়ে সংরক্ষিত PDF-র তালিকা দেখুন\n"
        "• `/status` দিয়ে সিস্টেম স্ট্যাটাস দেখুন",
        parse_mode="Markdown"
    )

async def handle_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ /file কমান্ডের মাধ্যমে PDF ফাইল হ্যান্ডেল করবে """
    if not update.message.document:
        await update.message.reply_text("❌ দয়া করে /file কমান্ডের সাথে একটি PDF ফাইল আপলোড করুন।")
        return
    
    document = update.message.document
    if not document.file_name.endswith('.pdf'):
        await update.message.reply_text("❌ শুধুমাত্র PDF ফাইল সমর্থিত।")
        return

    await update.message.reply_text(f"⏳ '{document.file_name}' প্রক্রিয়াকরণ শুরু হয়েছে...")
    
    try:
        file = await context.bot.get_file(document.file_id)
        pdf_bytes = await file.download_as_bytearray()
        
        full_text = extract_text_from_pdf_bytes(pdf_bytes)
        chunks = chunk_text(full_text)
        
        if not chunks:
            await update.message.reply_text("❌ PDF থেকে কোনো টেক্সট পাওয়া যায়নি।")
            return
        
        vector_count = save_to_pinecone(document.file_name, chunks)
        
        await update.message.reply_text(
            f"✅ **'{document.file_name}'** সফলভাবে প্রক্রিয়াকৃত হয়েছে!\n\n"
            f"📄 টেক্সট খণ্ড: {len(chunks)}\n"
            f"🗄️ Pinecone ভেক্টর: {vector_count}\n\n"
            f"ℹ️ এখন আপনি এই PDF সম্পর্কে প্রশ্ন করতে পারেন।",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"PDF প্রসেসিং ত্রুটি: {e}")
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ যেকোনো টেক্সট মেসেজকে প্রশ্ন হিসেবে গণ্য করে Gemma দিয়ে উত্তর দেবে """
    user_question = update.message.text
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        results = search_in_pinecone(user_question, top_k=3)
        
        if not results:
            await update.message.reply_text(
                "❌ কোনো প্রাসঙ্গিক তথ্য পাওয়া যায়নি। দয়া করে আগে `/file` দিয়ে PDF আপলোড করুন।",
                parse_mode="Markdown"
            )
            return
        
        answer = generate_answer_with_gemma(user_question, results)
        
        sources = "\n".join([f"• {r['filename']} (সাদৃশ্য: {r['score']:.2f})" for r in results])
        
        await update.message.reply_text(
            f"{answer}\n\n---\n📚 **সোর্স:**\n{sources}",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"প্রশ্ন প্রসেসিং ত্রুটি: {e}")
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def list_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ Pinecone-এ সংরক্ষিত ফাইলের তালিকা দেখাবে """
    try:
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
            await update.message.reply_text(f"📁 **সংরক্ষিত PDF ফাইল:**\n{file_list}", parse_mode="Markdown")
        else:
            await update.message.reply_text("ℹ️ এখনো কোনো PDF ফাইল সংরক্ষিত হয়নি।")
            
    except Exception as e:
        await update.message.reply_text(f"❌ ত্রুটি: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ সিস্টেম স্ট্যাটাস দেখাবে """
    hf_status = "✅ API Key সেট করা আছে" if HF_API_KEY else "❌ API Key নেই"
    status_text = f"""
🤖 **সিস্টেম স্ট্যাটাস**

**মডেল:** Gemma 4 E2B (Hugging Face Inference API)
**Pinecone:** সংযুক্ত ✅
**HF API:** {hf_status}
**ফ্রি টায়ার:** ৫০০০ রিকোয়েস্ট/মাস
"""
    await update.message.reply_text(status_text, parse_mode="Markdown")

# --- ৭. FastAPI Lifespan: Webhook সেটআপ ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bot অ্যাপ্লিকেশন তৈরি
    request = HTTPXRequest(connection_pool_size=10, read_timeout=60, write_timeout=60)
    bot = Bot(token=TELEGRAM_TOKEN, request=request)
    
    # ✅ সঠিক পদ্ধতি: তৈরি করা bot অবজেক্টটি ব্যবহার করুন
    ptb_app = Application.builder().bot(bot).build()
    
    # হ্যান্ডলার যুক্ত করুন
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("file", handle_file_command))
    ptb_app.add_handler(CommandHandler("list", list_files))
    ptb_app.add_handler(CommandHandler("status", status))
    ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_question))
    
    # বট ইনিশিয়ালাইজ করুন
    await ptb_app.initialize()
    
    # ✅ app.state-এ সংরক্ষণ করুন (সঠিক পদ্ধতি)
    app.state.ptb_app = ptb_app
    app.state.bot = bot
    
    # Webhook সেটআপ
    webhook_url = f"{RENDER_EXTERNAL_URL}/telegram-webhook"
    await bot.set_webhook(url=webhook_url, secret_token=SECRET_TOKEN)
    logger.info(f"✅ Webhook set to: {webhook_url}")
    
    yield  # ✅ শুধু yield, কোনো ডিকশনারি নয়
    
    # ক্লিনআপ: Webhook সরান ও বট শাটডাউন করুন
    await bot.delete_webhook()
    await ptb_app.shutdown()
    logger.info("👋 Bot shutdown complete")

# --- ৮. FastAPI অ্যাপ ---
app = FastAPI(lifespan=lifespan)

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """টেলিগ্রাম থেকে আসা আপডেট গ্রহণ করে।"""
    if request.headers.get('X-Telegram-Bot-Api-Secret-Token') != SECRET_TOKEN:
        return Response(status_code=403)
    
    # ✅ app.state থেকে ptb_app ও bot নিন (সঠিক পদ্ধতি)
    ptb_app = request.app.state.ptb_app
    bot = request.app.state.bot
    
    data = await request.json()
    update = Update.de_json(data, bot)
    
    asyncio.create_task(ptb_app.process_update(update))
    return Response(status_code=200)

@app.get("/healthcheck")
async def health():
    return {"status": "ok", "model": "Gemma 4 E2B via Hugging Face API"}

# --- ৯. মেইন এন্ট্রি পয়েন্ট ---
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        app,  # ✅ স্ট্রিং নয়, সরাসরি app অবজেক্ট
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
