#!/usr/bin/env python3
# pdf_to_pinecone_image_upload.py
"""
Internet Archive PDF Processor - Ultimate Production Grade
Complete Pinecone Upload with Namespace Strategy for All LLM Paradigms
No OpenAI Required - Local Embeddings + Full Training Readiness
Supports: SFT, DPO, PPO, RLHF, KTO, ORPO, SimPO, CPO, Agentic, Curriculum
"""

import os
import re
import time
import json
import tempfile
import ssl
import urllib.request
import urllib.error
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

import fitz  # PyMuPDF
from pymongo import MongoClient, errors as mongo_errors
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# ============ Logging Setup ============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/tafsir_processor.log')
    ]
)
logger = logging.getLogger(__name__)

# ============ Configuration ============

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "islamic_library")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "archive_links")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "islamic-knowledge")

MAX_PDFS_PER_RUN = int(os.environ.get("MAX_PDFS_PER_RUN", "50"))
PDF_BATCH_SIZE = int(os.environ.get("PDF_BATCH_SIZE", "50"))
PDF_SLEEP_BETWEEN = int(os.environ.get("PDF_SLEEP_BETWEEN", "1"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", "2"))

IMAGE_ZOOM = float(os.environ.get("IMAGE_ZOOM", "3.0"))
IMAGE_DPI = int(os.environ.get("IMAGE_DPI", "200"))
IMAGE_PREPROCESS = os.environ.get("IMAGE_PREPROCESS", "true").lower() == "true"

OCR_OEM = int(os.environ.get("OCR_OEM", "3"))
OCR_PSM = int(os.environ.get("OCR_PSM", "3"))
OCR_LANG = os.environ.get("OCR_LANG", "ben+ara+eng")
OCR_PRESERVE_SPACES = os.environ.get("OCR_PRESERVE_SPACES", "1")

PINECONE_BATCH_SIZE = int(os.environ.get("PINECONE_BATCH_SIZE", "100"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

TEMP_DIR = Path(os.environ.get("TEMP_DIR", "/tmp/tafsir_processor"))
TEMP_DIR.mkdir(exist_ok=True, parents=True)

ssl_context = ssl.create_default_context()

# ============ Enums ============

class BookCategory(str, Enum):
    TAFSIR = "tafsir"
    HADITH = "hadith"
    FIQH = "fiqh"
    AQIDAH = "aqidah"
    SEERAH = "seerah"
    HISTORY = "history"
    QURAN = "quran"
    GENERAL = "general"

class ContentType(str, Enum):
    AYAH = "ayah"
    HADITH = "hadith"
    TAFSIR = "tafsir"
    FIQH = "fiqh"
    QA = "qa"
    TABLE = "table"
    LIST = "list"
    GENERAL = "general"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

# ============ Memory Monitor ============

def get_memory_info() -> Dict:
    try:
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        return {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
            "process_mb": round(process.memory_info().rss / (1024**2), 1)
        }
    except:
        return {"error": "psutil not available"}

# ============ Image Preprocessor ============

class ImagePreprocessor:
    @staticmethod
    def preprocess(image: Image.Image) -> Image.Image:
        if not IMAGE_PREPROCESS:
            return image
        
        if image.mode != 'L':
            image = image.convert('L')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        
        return image

# ============ Text Cleaner ============

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u200b\u200c\u200d]', '', text)
        text = re.sub(r'([\u0600-\u06FF])\s+([\u064B-\u065F])', r'\1\2', text)
        text = re.sub(r'\s*([।,;:?!])\s*', r'\1 ', text)
        return text.strip()

# ============ Local Content Analyzer (No OpenAI) ============

class LocalContentAnalyzer:
    def __init__(self):
        self.bn_stopwords = {'এবং', 'ও', 'এর', 'যে', 'হয়', 'করে', 'একটি', 'কোন', 'না', 'থেকে'}
        self.ar_stopwords = {'في', 'من', 'على', 'أن', 'إن', 'كان', 'هذا', 'تلك', 'الذي', 'التي'}
    
    def analyze(self, text: str) -> Dict:
        cleaned = TextCleaner.clean(text)
        
        return {
            "content_type": self._detect_content_type(cleaned),
            "quality_score": self._calculate_quality(cleaned),
            "completeness_score": self._calculate_completeness(cleaned),
            "coherence_score": self._calculate_coherence(cleaned),
            "factual_score": self._calculate_factual_score(cleaned),
            "difficulty_level": self._estimate_difficulty(cleaned),
            "difficulty_score": self._calculate_difficulty_score(cleaned),
            "ready_for_sft": self._is_ready_for_sft(cleaned),
            "ready_for_preference": self._is_ready_for_preference(cleaned),
            "ready_for_rl": self._is_ready_for_rl(cleaned),
            "can_be_chosen": self._can_be_chosen(cleaned),
            "can_be_rejected": self._can_be_rejected(cleaned),
            "has_arabic": bool(re.search(r'[\u0600-\u06FF]', cleaned)),
            "has_bangla": bool(re.search(r'[\u0980-\u09FF]', cleaned)),
            "has_english": bool(re.search(r'[a-zA-Z]', cleaned)),
            "contains_ayah": bool(re.search(r'[ﷲ﷽]|آیة|سورة|আয়াত', cleaned)),
            "contains_hadith": bool(re.search(r'حديث|رواه|صحيح|হাদীস|বর্ণিত', cleaned)),
            "contains_reference": bool(re.search(r'\d+:\d+', cleaned)),
            "word_count": len(cleaned.split()),
            "char_count": len(cleaned),
            "cleaned_text": cleaned
        }
    
    def _detect_content_type(self, text: str) -> str:
        patterns = {
            "ayah": r'[ﷲ﷽]|آیة|سورة|আয়াত',
            "hadith": r'حديث|رواه|صحيح|হাদীস|বর্ণিত',
            "tafsir": r'تفسير|ব্যাখ্যা|অর্থাৎ|তাফসীর',
            "fiqh": r'حكم|হুকুম|মাসআলা|ফিকহ',
            "qa": r'\?|প্রশ্ন|উত্তর|سؤال|جواب',
            "table": r'\|.*\|.*\|',
            "list": r'^\s*[\d\-•*]',
        }
        for ctype, pattern in patterns.items():
            if re.search(pattern, text, re.MULTILINE):
                return ctype
        return "general"
    
    def _calculate_quality(self, text: str) -> float:
        score = 0.5
        length = len(text)
        if 200 < length < 800:
            score += 0.3
        elif 100 < length < 1000:
            score += 0.15
        if re.search(r'[।.!?]', text):
            score += 0.1
        if len(text.split('\n')) > 1:
            score += 0.1
        if re.search(r'\d+', text):
            score += 0.05
        return min(score, 1.0)
    
    def _calculate_completeness(self, text: str) -> float:
        length = len(text)
        if length < 50:
            return 0.3
        elif length < 150:
            return 0.6
        elif length < 400:
            return 0.8
        else:
            return 0.9
    
    def _calculate_coherence(self, text: str) -> float:
        score = 0.5
        if re.search(r'[।.!?]', text):
            score += 0.2
        if re.search(r'(?:অতএব|সুতরাং|কারণ|তাই|কেননা)', text):
            score += 0.2
        return min(score, 1.0)
    
    def _calculate_factual_score(self, text: str) -> float:
        score = 0.5
        if re.search(r'\d+:\d+', text):
            score += 0.2
        if re.search(r'[\u0600-\u06FF]', text):
            score += 0.15
        if re.search(r'(?:ইমাম|শাইখ|রহ\.|رح|قال)', text):
            score += 0.15
        return min(score, 1.0)
    
    def _estimate_difficulty(self, text: str) -> str:
        word_count = len(text.split())
        arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(text), 1)
        
        score = 0.0
        if word_count < 50:
            score += 0.1
        elif word_count < 150:
            score += 0.3
        elif word_count < 300:
            score += 0.5
        else:
            score += 0.7
        
        if arabic_ratio > 0.3:
            score += 0.2
        
        if score < 0.3:
            return "beginner"
        elif score < 0.5:
            return "intermediate"
        elif score < 0.7:
            return "advanced"
        else:
            return "expert"
    
    def _calculate_difficulty_score(self, text: str) -> float:
        word_count = len(text.split())
        arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(text), 1)
        return min(0.3 + (word_count / 500) * 0.4 + arabic_ratio * 0.3, 1.0)
    
    def _is_ready_for_sft(self, text: str) -> bool:
        quality = self._calculate_quality(text)
        length = len(text)
        return quality > 0.5 and 100 < length < 1000
    
    def _is_ready_for_preference(self, text: str) -> bool:
        quality = self._calculate_quality(text)
        coherence = self._calculate_coherence(text)
        return quality > 0.6 and coherence > 0.5 and len(text) > 100
    
    def _is_ready_for_rl(self, text: str) -> bool:
        quality = self._calculate_quality(text)
        factual = self._calculate_factual_score(text)
        return quality > 0.7 and factual > 0.6
    
    def _can_be_chosen(self, text: str) -> bool:
        return self._calculate_quality(text) > 0.7 and self._calculate_factual_score(text) > 0.6
    
    def _can_be_rejected(self, text: str) -> bool:
        return self._calculate_quality(text) < 0.4 or len(text) < 30

# ============ Local Embedding Provider (No OpenAI) ============

class LocalEmbeddingProvider:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
    
    def generate(self, text: str) -> List[float]:
        hash_functions = [hashlib.sha256, hashlib.sha512, hashlib.blake2b, hashlib.sha3_256]
        embedding = []
        text_bytes = text.encode('utf-8')
        
        for i in range(self.dimension):
            hash_func = hash_functions[i % len(hash_functions)]
            hash_obj = hash_func(text_bytes + str(i).encode())
            hash_bytes = hash_obj.digest()
            val = sum(hash_bytes[j % len(hash_bytes)] for j in range(3)) / (255.0 * 3)
            embedding.append(val)
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [v / norm for v in embedding]
        
        return embedding

# ============ Ultimate Pinecone Helper ============

class UltimatePineconeHelper:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.embedding_provider = LocalEmbeddingProvider(dimension=1536)
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        self.batch_vectors = {}
        self.stats = {"uploaded": 0, "failed": 0, "namespaces": set()}
        
        self.namespaces = {
            "tafsir": "তাফসীরের বই",
            "hadith": "হাদীসের বই",
            "fiqh": "ফিকহের বই",
            "aqidah": "আকীদার বই",
            "seerah": "সীরাহ/ইতিহাস",
            "quran": "কুরআন অনুবাদ",
            "general": "সাধারণ বই",
            "training_sft": "SFT ট্রেনিং ডেটা",
            "training_dpo": "DPO প্রিফারেন্স পেয়ার",
            "training_ppo": "PPO ট্রেনিং ডেটা",
            "training_rlhf": "RLHF ফিডব্যাক",
            "questions": "জেনারেটেড প্রশ্ন",
            "negative_samples": "নেগেটিভ স্যাম্পল",
            "curriculum_stage_1": "কারিকুলাম পর্যায় ১",
            "curriculum_stage_2": "কারিকুলাম পর্যায় ২",
            "curriculum_stage_3": "কারিকুলাম পর্যায় ৩",
            "curriculum_stage_4": "কারিকুলাম পর্যায় ৪",
            "curriculum_stage_5": "কারিকুলাম পর্যায় ৫",
            "high_quality": "উচ্চ মানের চাঙ্ক",
            "needs_review": "রিভিউ প্রয়োজন",
        }
    
    def _ensure_index_exists(self):
        try:
            existing = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)
            logger.info(f"Pinecone index ready: {self.index_name}")
        except Exception as e:
            logger.error(f"Pinecone error: {e}")
            raise
    
    def _determine_curriculum_stage(self, difficulty: str, quality: float, word_count: int) -> int:
        stage_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        base_stage = stage_map.get(difficulty, 2)
        if quality > 0.8 and word_count > 100:
            base_stage = min(base_stage + 1, 5)
        return base_stage
    
    def add_chunk(self, chunk_data: Dict) -> Optional[str]:
        chunk_id = chunk_data["chunk_id"]
        text = chunk_data.get("text", "")
        book_category = chunk_data.get("book_category", "general")
        quality = chunk_data.get("quality_score", 0.5)
        difficulty = chunk_data.get("difficulty_level", "intermediate")
        word_count = chunk_data.get("word_count", 0)
        
        # Determine namespaces
        namespaces = [book_category]
        
        if quality >= 0.8:
            namespaces.append("high_quality")
        elif quality < 0.4:
            namespaces.append("needs_review")
        
        if chunk_data.get("ready_for_sft"):
            namespaces.append("training_sft")
        if chunk_data.get("ready_for_preference"):
            namespaces.append("training_dpo")
        if chunk_data.get("ready_for_rl"):
            namespaces.append("training_ppo")
            namespaces.append("training_rlhf")
        
        if chunk_data.get("can_be_rejected"):
            namespaces.append("negative_samples")
        
        curriculum_stage = self._determine_curriculum_stage(difficulty, quality, word_count)
        namespaces.append(f"curriculum_stage_{curriculum_stage}")
        
        # Build metadata
        metadata = {
            "chunk_id": chunk_id,
            "book_id": chunk_data.get("book_id", "unknown"),
            "book_name": chunk_data.get("book_name", "")[:100],
            "book_category": book_category,
            "volume": chunk_data.get("volume", 1),
            "page_number": chunk_data.get("page_number", 0),
            "chunk_index": chunk_data.get("chunk_index", 0),
            "content_type": chunk_data.get("content_type", "general"),
            "char_count": chunk_data.get("char_count", 0),
            "word_count": word_count,
            "has_arabic": 1 if chunk_data.get("has_arabic") else 0,
            "has_bangla": 1 if chunk_data.get("has_bangla") else 0,
            "contains_ayah": 1 if chunk_data.get("contains_ayah") else 0,
            "contains_hadith": 1 if chunk_data.get("contains_hadith") else 0,
            "contains_reference": 1 if chunk_data.get("contains_reference") else 0,
            "quality_score": int(quality * 100),
            "completeness_score": int(chunk_data.get("completeness_score", 0.5) * 100),
            "coherence_score": int(chunk_data.get("coherence_score", 0.5) * 100),
            "factual_score": int(chunk_data.get("factual_score", 0.5) * 100),
            "difficulty_level": difficulty,
            "difficulty_score": int(chunk_data.get("difficulty_score", 0.5) * 100),
            "ready_for_sft": 1 if chunk_data.get("ready_for_sft") else 0,
            "ready_for_dpo": 1 if chunk_data.get("ready_for_preference") else 0,
            "ready_for_ppo": 1 if chunk_data.get("ready_for_rl") else 0,
            "ready_for_rlhf": 1 if chunk_data.get("ready_for_rl") else 0,
            "can_be_chosen": 1 if chunk_data.get("can_be_chosen") else 0,
            "can_be_rejected": 1 if chunk_data.get("can_be_rejected") else 0,
            "curriculum_stage": curriculum_stage,
            "text_preview": text[:500],
            "version": "3.0",
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Generate embedding
        embedding = self.embedding_provider.generate(text)
        
        # Add to batch for each namespace
        for namespace in namespaces:
            if namespace not in self.batch_vectors:
                self.batch_vectors[namespace] = []
                self.stats["namespaces"].add(namespace)
            
            self.batch_vectors[namespace].append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })
        
        if self.should_flush():
            self.flush_all()
        
        return chunk_id
    
    def should_flush(self) -> bool:
        total = sum(len(v) for v in self.batch_vectors.values())
        return total >= PINECONE_BATCH_SIZE
    
    def flush_namespace(self, namespace: str) -> int:
        if namespace not in self.batch_vectors or not self.batch_vectors[namespace]:
            return 0
        
        try:
            vectors = self.batch_vectors[namespace]
            for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
                batch = vectors[i:i+PINECONE_BATCH_SIZE]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            count = len(vectors)
            self.stats["uploaded"] += count
            logger.info(f"Uploaded {count} vectors to '{namespace}'")
            self.batch_vectors[namespace] = []
            return count
        except Exception as e:
            self.stats["failed"] += len(vectors)
            logger.error(f"Upload failed for '{namespace}': {e}")
            return 0
    
    def flush_all(self):
        total = 0
        for namespace in list(self.batch_vectors.keys()):
            total += self.flush_namespace(namespace)
        return total
    
    def close(self):
        self.flush_all()
        logger.info(f"Pinecone stats: {self.stats}")

# ============ MongoDB Helper ============

class MongoDBHelper:
    def __init__(self):
        self._connect()
        self._init_collections()
    
    def _connect(self):
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                self.client = MongoClient(MONGODB_URI, maxPoolSize=10, serverSelectionTimeoutMS=5000)
                self.client.admin.command('ping')
                self.db = self.client[MONGODB_DB]
                self.collection = self.db[MONGODB_COLLECTION]
                logger.info(f"Connected to MongoDB: {MONGODB_DB}")
                return
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning(f"MongoDB connection attempt {attempt+1} failed: {e}")
    
    def _init_collections(self):
        collections = ["books", "chunks", "pages", "training_queue", "feedback", "metrics"]
        for coll in collections:
            if coll not in self.db.list_collection_names():
                self.db.create_collection(coll)
        
        self.db["chunks"].create_index("chunk_id", unique=True)
        self.db["chunks"].create_index("book_id")
        self.db["chunks"].create_index("quality_score")
    
    def get_pending_archives(self) -> List[Dict]:
        query = {
            "status": {"$in": ["pending", "failed"]},
            "retry_count": {"$lt": MAX_RETRIES}
        }
        return list(self.collection.find(query).sort([
            ("priority", -1), 
            ("created_at", 1)
        ]).limit(MAX_PDFS_PER_RUN))
    
    def update_status(self, archive_id: str, status: str, **kwargs):
        update_data = {"status": status, "updated_at": datetime.utcnow(), **kwargs}
        self.collection.update_one({"_id": archive_id}, {"$set": update_data})
    
    def increment_retry(self, archive_id: str):
        self.collection.update_one(
            {"_id": archive_id},
            {"$inc": {"retry_count": 1}, "$set": {"updated_at": datetime.utcnow()}}
        )
    
    def save_book_metadata(self, book_metadata: Dict):
        self.db["books"].update_one(
            {"book_id": book_metadata["book_id"]},
            {"$set": book_metadata},
            upsert=True
        )
    
    def save_chunk(self, chunk: Dict):
        try:
            self.db["chunks"].update_one(
                {"chunk_id": chunk["chunk_id"]},
                {"$set": chunk},
                upsert=True
            )
        except mongo_errors.DuplicateKeyError:
            pass
    
    def save_page(self, page_data: Dict):
        self.db["pages"].update_one(
            {"page_id": page_data["page_id"]},
            {"$set": page_data},
            upsert=True
        )
    
    def close(self):
        if hasattr(self, 'client'):
            self.client.close()

# ============ OCR Processor ============

class OCRProcessor:
    def __init__(self, pinecone_helper: UltimatePineconeHelper, mongo_helper: MongoDBHelper):
        self.pinecone = pinecone_helper
        self.mongo = mongo_helper
        self.analyzer = LocalContentAnalyzer()
        self.preprocessor = ImagePreprocessor()
        self._check_tesseract()
    
    def _check_tesseract(self):
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise
    
    def _split_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        if not words:
            return []
        
        i = 0
        while i < len(words):
            chunk_words = []
            current_len = 0
            for j in range(i, len(words)):
                word_len = len(words[j]) + 1
                if current_len + word_len > CHUNK_SIZE and chunk_words:
                    break
                chunk_words.append(words[j])
                current_len += word_len
            if chunk_words:
                chunks.append(' '.join(chunk_words))
            overlap_words = max(1, int(len(chunk_words) * CHUNK_OVERLAP / CHUNK_SIZE))
            i += max(1, len(chunk_words) - overlap_words)
        return chunks
    
    def process_page(self, image_path: Path, metadata: Dict, book_metadata: Dict,
                     page_num: int, total_pages: int) -> Dict:
        try:
            img = Image.open(image_path)
            img = self.preprocessor.preprocess(img)
            
            custom_config = f'--oem {OCR_OEM} --psm {OCR_PSM} -l {OCR_LANG} --dpi {IMAGE_DPI} -c preserve_interword_spaces={OCR_PRESERVE_SPACES}'
            text = pytesseract.image_to_string(img, config=custom_config)
            text = TextCleaner.clean(text)
            
            if not text:
                return {"page_num": page_num, "chunks": [], "text": "", "char_count": 0}
            
            analysis = self.analyzer.analyze(text)
            chunks = self._split_text(analysis["cleaned_text"])
            chunk_data = []
            
            page_id = f"{book_metadata['book_id']}_v{metadata.get('volume', 1)}_p{page_num}"
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = hashlib.md5(f"{page_id}_c{chunk_idx}".encode()).hexdigest()
                
                chunk = {
                    "chunk_id": chunk_id,
                    "book_id": book_metadata["book_id"],
                    "book_name": book_metadata["book_name"],
                    "book_category": book_metadata.get("category", "general"),
                    "volume": metadata.get("volume", 1),
                    "pdf_number": metadata.get("pdf_number", 1),
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "text": chunk_text,
                    **{k: v for k, v in analysis.items() if k != "cleaned_text"}
                }
                
                self.pinecone.add_chunk(chunk)
                self.mongo.save_chunk(chunk)
                chunk_data.append({"chunk_id": chunk_id, "text_length": len(chunk_text)})
            
            # Save page metadata
            self.mongo.save_page({
                "page_id": page_id,
                "book_id": book_metadata["book_id"],
                "page_num": page_num,
                "total_pages": total_pages,
                "full_text": text,
                "char_count": len(text),
                "chunk_ids": [c["chunk_id"] for c in chunk_data],
                "processed_at": datetime.utcnow().isoformat()
            })
            
            return {"page_num": page_num, "chunks": chunk_data, "text": text, "char_count": len(text)}
            
        except Exception as e:
            logger.error(f"OCR error page {page_num}: {e}")
            return {"page_num": page_num, "chunks": [], "text": "", "error": str(e)}

# ============ PDF Downloader ============

class PDFDownloader:
    @staticmethod
    def download_parallel(urls: List[str], max_workers: int = MAX_WORKERS) -> Dict[str, str]:
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(PDFDownloader._download_single, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    logger.error(f"Download failed {url[:60]}... : {e}")
                    results[url] = None
        return results
    
    @staticmethod
    def _download_single(url: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR)
                headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/pdf'}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
                    temp_file.write(response.read())
                temp_file.close()
                return temp_file.name
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
        raise Exception(f"Failed after {MAX_RETRIES} attempts")

# ============ Main Processor ============

class TafsirProcessor:
    def __init__(self):
        self.mongo = MongoDBHelper()
        self.pinecone = UltimatePineconeHelper()
        self.ocr = OCRProcessor(self.pinecone, self.mongo)
        
        mem_info = get_memory_info()
        logger.info(f"RAM: {mem_info.get('total_gb', 'N/A')} GB total, {mem_info.get('available_gb', 'N/A')} GB available")
    
    def process_archive(self, archive_item: Dict) -> bool:
        archive_id = archive_item["_id"]
        archive_url = archive_item["url"]
        book_name = archive_item.get("book_name", f"tafsir_{archive_id}")
        
        book_metadata = archive_item.get("book_metadata", {
            "book_id": hashlib.md5(book_name.encode()).hexdigest()[:12],
            "book_name": book_name,
            "category": BookCategory.TAFSIR.value,
            "namespace": BookCategory.TAFSIR.value,
            "language": "mixed"
        })
        
        settings = archive_item.get("processing_settings", {})
        pdf_batch = settings.get("pdf_batch_size", PDF_BATCH_SIZE)
        max_pdfs = settings.get("max_pdfs_per_run", MAX_PDFS_PER_RUN)
        zoom = settings.get("image_zoom", IMAGE_ZOOM)
        dpi = settings.get("image_dpi", IMAGE_DPI)
        workers = settings.get("max_workers", MAX_WORKERS)
        ocr_lang = settings.get("ocr_lang", OCR_LANG)
        ocr_workers = settings.get("ocr_workers", OCR_WORKERS)

        logger.info(f"Processing: {book_name}")
        logger.info(f"Category: {book_metadata.get('category', 'general')}")
        logger.info(f"Settings: Batch={pdf_batch}, Zoom={zoom}, DPI={dpi}, OCR={ocr_lang}")

        try:
            self.mongo.update_status(archive_id, "processing", started_at=datetime.utcnow())
            self.mongo.save_book_metadata(book_metadata)

            pdf_urls = self._generate_pdf_urls(archive_url)
            if not pdf_urls:
                self.mongo.update_status(archive_id, "failed", error="No PDFs found")
                return False

            if len(pdf_urls) > max_pdfs:
                logger.info(f"Limiting PDFs from {len(pdf_urls)} to {max_pdfs}")
                pdf_urls = pdf_urls[:max_pdfs]

            logger.info(f"Found {len(pdf_urls)} PDFs")
            download_results = PDFDownloader.download_parallel(pdf_urls, max_workers=workers)

            success_count = 0
            total_chars = 0
            total_chunks = 0

            for idx, url in enumerate(pdf_urls):
                pdf_number = idx + 1
                pdf_path = download_results.get(url)
                if not pdf_path:
                    continue

                logger.info(f"PDF {pdf_number}/{len(pdf_urls)}")
                
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                
                for batch_start in range(0, total_pages, pdf_batch):
                    batch_end = min(batch_start + pdf_batch, total_pages)
                    batch_folder = Path(tempfile.mkdtemp(prefix=f"pdf{pdf_number}_batch_", dir=TEMP_DIR))
                    
                    try:
                        batch_doc = fitz.open(pdf_path)
                        for page_num in range(batch_start, batch_end):
                            page = batch_doc.load_page(page_num)
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page.get_pixmap(matrix=mat, alpha=False)
                            img_path = batch_folder / f"page_{page_num+1:04d}.png"
                            pix.save(img_path, "png")
                            
                            metadata = {"volume": pdf_number, "pdf_number": pdf_number}
                            result = self.ocr.process_page(
                                img_path, metadata, book_metadata, page_num+1, total_pages
                            )
                            
                            total_chars += result.get("char_count", 0)
                            total_chunks += len(result.get("chunks", []))
                        
                        batch_doc.close()
                        logger.info(f"PDF {pdf_number}: Pages {batch_start+1}-{batch_end} complete")
                        
                    finally:
                        shutil.rmtree(batch_folder, ignore_errors=True)
                
                doc.close()
                try:
                    os.unlink(pdf_path)
                except:
                    pass
                
                success_count += 1
                self.mongo.update_status(
                    archive_id, "processing",
                    current_pdf=pdf_number, total_pdfs=len(pdf_urls),
                    completed_pdfs=success_count, total_chars=total_chars,
                    total_chunks=total_chunks
                )
                
                if idx < len(pdf_urls) - 1:
                    time.sleep(PDF_SLEEP_BETWEEN)

            self.pinecone.flush_all()
            
            final_status = "completed" if success_count == len(pdf_urls) else "partial"
            self.mongo.update_status(
                archive_id, final_status,
                completed_at=datetime.utcnow(),
                total_pdfs=len(pdf_urls), completed_pdfs=success_count,
                total_chars=total_chars, total_chunks=total_chunks
            )

            logger.info(f"Complete: {success_count}/{len(pdf_urls)} PDFs, {total_chars} chars, {total_chunks} chunks")
            return success_count > 0

        except Exception as e:
            logger.error(f"Archive error: {e}")
            self.mongo.update_status(archive_id, "failed", error=str(e))
            self.mongo.increment_retry(archive_id)
            return False
    
    def _generate_pdf_urls(self, archive_url: str) -> List[str]:
        if '/details/' in archive_url:
            item_id = archive_url.split('/details/')[-1].split('/')[0]
        elif '/download/' in archive_url:
            item_id = archive_url.split('/download/')[-1].split('/')[0]
        else:
            return []
        logger.info(f"Item ID: {item_id}")
        return [f"https://archive.org/download/{item_id}/{i}.pdf" for i in range(1, 51)]
    
    def run(self):
        logger.info(f"Starting Tafsir OCR Processor")
        logger.info(f"MongoDB: {MONGODB_DB}, Pinecone: {PINECONE_INDEX_NAME}")
        
        pending = self.mongo.get_pending_archives()
        if not pending:
            logger.info("No pending archives")
            return

        logger.info(f"Found {len(pending)} pending archives")
        for idx, archive in enumerate(pending):
            logger.info(f"Archive {idx+1}/{len(pending)}")
            self.process_archive(archive)
            if idx < len(pending) - 1:
                time.sleep(2)

        logger.info("All processing complete!")
        self.mongo.close()
        self.pinecone.close()

# ============ Entry Point ============

def main():
    required = ["MONGODB_URI", "PINECONE_API_KEY"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing: {', '.join(missing)}")
        exit(1)

    logger.info(f"Runner: {os.environ.get('RUNNER_OS', 'Unknown')}")
    TafsirProcessor().run()

if __name__ == "__main__":
    main()