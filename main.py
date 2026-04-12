import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET", "my_super_secret_token_2026")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 বট চালু আছে! /help লিখুন।")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📚 এটি একটি টেস্ট বট। /start - চালু করুন")

@asynccontextmanager
async def lifespan(app: FastAPI):
    request = HTTPXRequest(read_timeout=60, write_timeout=60)
    bot = Bot(token=TELEGRAM_TOKEN, request=request)
    ptb_app = Application.builder().bot(bot).build()
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("help", help_command))
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

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    if request.headers.get('X-Telegram-Bot-Api-Secret-Token') != SECRET_TOKEN:
        return Response(status_code=403)
    ptb_app = request.app.state.ptb_app
    bot = request.app.state.bot
    data = await request.json()
    update = Update.de_json(data, bot)
    asyncio.create_task(ptb_app.process_update(update))
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
