#!/usr/bin/env python3
"""
Internet Archive PDF Processor - GitHub Actions Optimized
Full utilization of 7GB RAM for maximum throughput
Supports per-archive processing settings from MongoDB
OCR with Tesseract and save to Pinecone
"""

import os
import re
import time
import json
import tempfile
import gc
import ssl
import random
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import hashlib

import fitz  # PyMuPDF
from pymongo import MongoClient
from PIL import Image
import pytesseract
from pinecone import Pinecone, ServerlessSpec

# ============ GitHub Actions Optimized Configuration ============

# GitHub Actions provides 7GB RAM - Let's use it efficiently!
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "tafsir_db")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "archive_links")

# Pinecone Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "tafsir-ocr")

# Optimized settings for GitHub Actions (7GB RAM) - Default fallback values
MAX_PDFS_PER_RUN = int(os.environ.get("MAX_PDFS_PER_RUN", "50"))
PDF_BATCH_SIZE = int(os.environ.get("PDF_BATCH_SIZE", "50"))
PDF_SLEEP_BETWEEN = int(os.environ.get("PDF_SLEEP_BETWEEN", "1"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Parallel processing settings
MAX_PARALLEL_PDFS = int(os.environ.get("MAX_PARALLEL_PDFS", "2"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", "2"))

# High-quality image settings (RAM allows it)
IMAGE_ZOOM = float(os.environ.get("IMAGE_ZOOM", "3.0"))
IMAGE_DPI = int(os.environ.get("IMAGE_DPI", "200"))

# OCR Settings (Best Quality)
OCR_OEM = int(os.environ.get("OCR_OEM", "3"))
OCR_PSM = int(os.environ.get("OCR_PSM", "3"))
OCR_LANG = os.environ.get("OCR_LANG", "ben")
OCR_PRESERVE_SPACES = os.environ.get("OCR_PRESERVE_SPACES", "1")

# Pinecone Batch Settings
PINECONE_BATCH_SIZE = int(os.environ.get("PINECONE_BATCH_SIZE", "100"))

# Temp Directory
TEMP_DIR = Path(os.environ.get("TEMP_DIR", "/tmp/tafsir_processor"))
TEMP_DIR.mkdir(exist_ok=True, parents=True)

ssl_context = ssl.create_default_context()

# ============ Memory Monitor ============

def get_memory_info():
    """Get memory usage information"""
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

# ============ MongoDB Helper ============

class MongoDBHelper:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI, maxPoolSize=10)
        self.db = self.client[MONGODB_DB]
        self.collection = self.db[MONGODB_COLLECTION]

    def get_pending_archives(self, limit: int = MAX_PDFS_PER_RUN) -> List[Dict]:
        """Fetch pending archives with priority"""
        query = {
            "status": {"$in": ["pending", "failed"]},
            "retry_count": {"$lt": MAX_RETRIES}
        }

        archives = list(self.collection.find(query).sort([
            ("priority", -1), 
            ("created_at", 1)
        ]).limit(limit))

        for archive in archives:
            if "processing_settings" not in archive:
                archive["processing_settings"] = {
                    "pdf_batch_size": PDF_BATCH_SIZE,
                    "max_pdfs_per_run": MAX_PDFS_PER_RUN,
                    "image_zoom": IMAGE_ZOOM,
                    "image_dpi": IMAGE_DPI,
                    "max_parallel_pdfs": MAX_PARALLEL_PDFS,
                    "max_workers": MAX_WORKERS,
                    "ocr_oem": OCR_OEM,
                    "ocr_psm": OCR_PSM,
                    "ocr_lang": OCR_LANG,
                    "ocr_workers": OCR_WORKERS
                }

        return archives

    def update_status(self, archive_id: str, status: str, **kwargs):
        """Update processing status"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow(),
            **kwargs
        }

        self.collection.update_one(
            {"_id": archive_id},
            {"$set": update_data}
        )

    def increment_retry(self, archive_id: str):
        """Increment retry count"""
        self.collection.update_one(
            {"_id": archive_id},
            {
                "$inc": {"retry_count": 1},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    def save_ocr_progress(self, archive_id: str, pdf_number: int, page_num: int, 
                          text_length: int, pinecone_ids: List[str]):
        """Save OCR progress"""
        self.collection.update_one(
            {"_id": archive_id},
            {
                "$set": {
                    f"ocr_progress.{pdf_number}.{page_num}": {
                        "text_length": text_length,
                        "pinecone_ids": pinecone_ids,
                        "processed_at": datetime.utcnow()
                    },
                    "updated_at": datetime.utcnow()
                }
            }
        )

    def close(self):
        self.client.close()

# ============ Pinecone Helper (Updated for v6+) ============

class PineconeHelper:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.dimension = 1536
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        self.batch_vectors = []

    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                print(f"[Pinecone] Creating index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)
                print(f"[Pinecone] Index created and ready")
            else:
                print(f"[Pinecone] Index found: {self.index_name}")
        except Exception as e:
            print(f"[Pinecone] Error checking index: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using hash-based method"""
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        embedding = []
        for i in range(self.dimension):
            val = hash_bytes[i % len(hash_bytes)] / 255.0
            embedding.append(val)
        return embedding

    def add_text(self, text: str, metadata: Dict) -> str:
        """Add text to batch for Pinecone upsert"""
        vector_id = hashlib.md5(f"{metadata.get('archive_id')}_{metadata.get('pdf_number')}_{metadata.get('page_num')}_{metadata.get('chunk_index')}".encode()).hexdigest()
        
        embedding = self.generate_embedding(text)
        
        self.batch_vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": text[:1000],
                **metadata
            }
        })
        
        return vector_id

    def flush_batch(self) -> int:
        """Upload batch to Pinecone"""
        if not self.batch_vectors:
            return 0
        
        try:
            self.index.upsert(vectors=self.batch_vectors)
            count = len(self.batch_vectors)
            print(f"[Pinecone] Uploaded {count} vectors")
            self.batch_vectors = []
            return count
        except Exception as e:
            print(f"[Pinecone] Upload failed: {e}")
            return 0

    def should_flush(self) -> bool:
        """Check if batch should be flushed"""
        return len(self.batch_vectors) >= PINECONE_BATCH_SIZE

    def close(self):
        """Flush remaining vectors"""
        if self.batch_vectors:
            self.flush_batch()

# ============ OCR Processor ============

class OCRProcessor:
    def __init__(self, pinecone_helper: PineconeHelper):
        self.pinecone = pinecone_helper
        
        try:
            pytesseract.get_tesseract_version()
            print(f"[OCR] Tesseract version: {pytesseract.get_tesseract_version()}")
        except:
            print("[OCR] Tesseract not found. Please install tesseract-ocr")
            raise

    def process_image(self, image_path: Path, metadata: Dict, 
                      ocr_oem: int = OCR_OEM,
                      ocr_psm: int = OCR_PSM,
                      ocr_lang: str = OCR_LANG,
                      ocr_dpi: int = IMAGE_DPI) -> Tuple[str, List[str]]:
        """Process single image with OCR and save to Pinecone"""
        
        try:
            img = Image.open(image_path)
            
            custom_config = f'--oem {ocr_oem} --psm {ocr_psm} -l {ocr_lang} --dpi {ocr_dpi} -c preserve_interword_spaces={OCR_PRESERVE_SPACES}'
            
            text = pytesseract.image_to_string(img, config=custom_config)
            text = text.strip()
            
            if not text:
                return "", []
            
            chunks = self._split_text(text, chunk_size=500)
            pinecone_ids = []
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk)
                }
                
                vector_id = self.pinecone.add_text(chunk, chunk_metadata)
                pinecone_ids.append(vector_id)
            
            if self.pinecone.should_flush():
                self.pinecone.flush_batch()
            
            return text, pinecone_ids
            
        except Exception as e:
            print(f"[OCR] Error processing {image_path.name}: {e}")
            return "", []

    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_length + word_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_images_parallel(self, image_paths: List[Path], metadata_list: List[Dict],
                                 ocr_oem: int = OCR_OEM,
                                 ocr_psm: int = OCR_PSM,
                                 ocr_lang: str = OCR_LANG,
                                 ocr_dpi: int = IMAGE_DPI,
                                 max_workers: int = OCR_WORKERS) -> List[Dict]:
        """Process multiple images in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    self.process_image, 
                    path, 
                    meta, 
                    ocr_oem, 
                    ocr_psm, 
                    ocr_lang, 
                    ocr_dpi
                ): (path, meta) 
                for path, meta in zip(image_paths, metadata_list)
            }
            
            for future in as_completed(future_to_image):
                image_path, metadata = future_to_image[future]
                try:
                    text, pinecone_ids = future.result()
                    results.append({
                        "image": str(image_path),
                        "text_length": len(text),
                        "pinecone_ids": pinecone_ids,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"[OCR] Failed {image_path.name}: {e}")
                    results.append({
                        "image": str(image_path),
                        "text_length": 0,
                        "pinecone_ids": [],
                        "error": str(e),
                        "metadata": metadata
                    })
        
        return results

# ============ Optimized PDF Downloader ============

class PDFDownloader:
    @staticmethod
    def download_parallel(urls: List[str], max_workers: int = MAX_WORKERS) -> Dict[str, str]:
        """Download multiple PDFs in parallel"""
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(PDFDownloader._download_single, url): url 
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    pdf_path = future.result()
                    results[url] = pdf_path
                except Exception as e:
                    print(f"[Download] Failed {url[:60]}... : {e}")
                    results[url] = None

        return results

    @staticmethod
    def _download_single(url: str) -> str:
        """Download single PDF"""
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.pdf', 
                    dir=TEMP_DIR
                )

                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                    'Accept': 'application/pdf,application/octet-stream,*/*'
                }

                req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
                    temp_file.write(response.read())

                temp_file.close()
                return temp_file.name

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise

        raise Exception(f"Failed after {MAX_RETRIES} attempts")

# ============ Optimized PDF to OCR Processor ============

class PDFToOCRProcessor:
    def __init__(self, ocr_processor: OCRProcessor, mongo_helper: MongoDBHelper):
        self.ocr = ocr_processor
        self.mongo = mongo_helper

    def process_pdf_to_ocr(self, pdf_url: str, pdf_path: str, pdf_number: int, 
                           folder_name: str, archive_id: str,
                           pdf_batch_size: int = PDF_BATCH_SIZE,
                           image_zoom: float = IMAGE_ZOOM,
                           image_dpi: int = IMAGE_DPI,
                           ocr_oem: int = OCR_OEM,
                           ocr_psm: int = OCR_PSM,
                           ocr_lang: str = OCR_LANG,
                           ocr_workers: int = OCR_WORKERS) -> Dict:
        """Process PDF: convert to PNG, run OCR, save to Pinecone"""

        result = {
            "pdf_number": pdf_number,
            "status": "pending",
            "pages_processed": 0,
            "total_text_chars": 0,
            "pinecone_vectors": 0
        }

        clean_folder = self._clean_folder_name(folder_name)

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            print(f"[PDF {pdf_number}] {total_pages} pages total (Batch={pdf_batch_size}, Zoom={image_zoom}, DPI={image_dpi})")
            print(f"[PDF {pdf_number}] OCR Settings: OEM={ocr_oem}, PSM={ocr_psm}, Lang={ocr_lang}, Workers={ocr_workers}")

            pages_processed = 0
            total_text_chars = 0
            total_pinecone_vectors = 0

            for batch_start in range(0, total_pages, pdf_batch_size):
                batch_end = min(batch_start + pdf_batch_size, total_pages)
                batch_number = (batch_start // pdf_batch_size) + 1

                batch_folder = Path(tempfile.mkdtemp(
                    prefix=f"pdf{pdf_number}_batch{batch_number}_", 
                    dir=TEMP_DIR
                ))

                try:
                    batch_doc = fitz.open(pdf_path)
                    
                    image_paths = []
                    metadata_list = []
                    
                    for page_num in range(batch_start, batch_end):
                        page = batch_doc.load_page(page_num)
                        
                        mat = fitz.Matrix(image_zoom, image_zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)

                        img_name = f"page_{page_num+1:04d}.png"
                        img_path = batch_folder / img_name
                        pix.save(img_path, "png")

                        image_paths.append(img_path)
                        metadata_list.append({
                            "archive_id": archive_id,
                            "pdf_number": pdf_number,
                            "page_num": page_num + 1,
                            "total_pages": total_pages,
                            "folder_name": clean_folder,
                            "batch_number": batch_number,
                            "image_zoom": image_zoom,
                            "image_dpi": image_dpi
                        })

                        pages_processed += 1
                        pix = None
                        page = None

                    batch_doc.close()

                    if image_paths:
                        print(f"[PDF {pdf_number}] Batch {batch_number}: Running OCR on {len(image_paths)} images...")
                        
                        ocr_results = self.ocr.process_images_parallel(
                            image_paths, 
                            metadata_list,
                            ocr_oem=ocr_oem,
                            ocr_psm=ocr_psm,
                            ocr_lang=ocr_lang,
                            ocr_dpi=image_dpi,
                            max_workers=ocr_workers
                        )
                        
                        batch_chars = sum(r["text_length"] for r in ocr_results)
                        batch_vectors = sum(len(r["pinecone_ids"]) for r in ocr_results)
                        
                        total_text_chars += batch_chars
                        total_pinecone_vectors += batch_vectors
                        
                        print(f"[PDF {pdf_number}] Batch {batch_number} complete: {len(image_paths)} pages, {batch_chars} chars, {batch_vectors} vectors")
                        
                        for r in ocr_results:
                            meta = r["metadata"]
                            self.mongo.save_ocr_progress(
                                archive_id,
                                pdf_number,
                                meta["page_num"],
                                r["text_length"],
                                r["pinecone_ids"]
                            )

                finally:
                    try:
                        if batch_folder.exists():
                            import shutil
                            shutil.rmtree(batch_folder, ignore_errors=True)
                    except:
                        pass

                if batch_end < total_pages:
                    time.sleep(0.5)

            self.ocr.pinecone.flush_batch()

            result["status"] = "completed"
            result["pages_processed"] = pages_processed
            result["total_text_chars"] = total_text_chars
            result["pinecone_vectors"] = total_pinecone_vectors

        except Exception as e:
            print(f"[PDF {pdf_number}] Error: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    @staticmethod
    def _clean_folder_name(name: str) -> str:
        """Clean folder name"""
        import re
        return re.sub(r'[^\w\-_]', '_', name.replace(' ', '_'))

# ============ Main Processor ============

class TafsirProcessor:
    def __init__(self):
        self.mongo = MongoDBHelper()
        self.pinecone = PineconeHelper()
        self.ocr = OCRProcessor(self.pinecone)
        self.pdf_processor = PDFToOCRProcessor(self.ocr, self.mongo)

        mem_info = get_memory_info()
        print(f"[System] RAM: {mem_info.get('total_gb', 'N/A')} GB total, "
              f"{mem_info.get('available_gb', 'N/A')} GB available")

    def process_archive_optimized(self, archive_item: Dict) -> bool:
        """Process archive with parallel downloads and OCR to Pinecone"""

        archive_id = archive_item["_id"]
        archive_url = archive_item["url"]
        book_name = archive_item.get("book_name", f"tafsir_{archive_id}")

        settings = archive_item.get("processing_settings", {})
        archive_pdf_batch_size = settings.get("pdf_batch_size", PDF_BATCH_SIZE)
        archive_max_pdfs_per_run = settings.get("max_pdfs_per_run", MAX_PDFS_PER_RUN)
        archive_image_zoom = settings.get("image_zoom", IMAGE_ZOOM)
        archive_image_dpi = settings.get("image_dpi", IMAGE_DPI)
        archive_max_workers = settings.get("max_workers", MAX_WORKERS)
        archive_ocr_oem = settings.get("ocr_oem", OCR_OEM)
        archive_ocr_psm = settings.get("ocr_psm", OCR_PSM)
        archive_ocr_lang = settings.get("ocr_lang", OCR_LANG)
        archive_ocr_workers = settings.get("ocr_workers", OCR_WORKERS)

        print(f"\n{'='*70}")
        print(f"[Archive] {book_name}")
        print(f"[Archive] URL: {archive_url}")
        print(f"[Archive] PDF Settings: Batch={archive_pdf_batch_size}, Zoom={archive_image_zoom}, DPI={archive_image_dpi}")
        print(f"[Archive] OCR Settings: OEM={archive_ocr_oem}, PSM={archive_ocr_psm}, Lang={archive_ocr_lang}, Workers={archive_ocr_workers}")
        print(f"{'='*70}\n")

        try:
            self.mongo.update_status(archive_id, "processing", started_at=datetime.utcnow())

            pdf_urls = self._generate_pdf_urls_optimized(archive_url)

            if not pdf_urls:
                print(f"[Archive] No PDFs found")
                self.mongo.update_status(archive_id, "failed", error="No PDFs found")
                return False

            if len(pdf_urls) > archive_max_pdfs_per_run:
                print(f"[Archive] Limiting PDFs from {len(pdf_urls)} to {archive_max_pdfs_per_run}")
                pdf_urls = pdf_urls[:archive_max_pdfs_per_run]

            print(f"[Archive] Found {len(pdf_urls)} PDFs to process")

            print(f"[Archive] Downloading {len(pdf_urls)} PDFs in parallel (workers={archive_max_workers})...")
            download_results = PDFDownloader.download_parallel(pdf_urls, max_workers=archive_max_workers)

            all_results = []
            success_count = 0
            total_chars = 0
            total_vectors = 0

            for idx, url in enumerate(pdf_urls):
                pdf_number = idx + 1
                pdf_path = download_results.get(url)

                if not pdf_path:
                    print(f"[Archive] PDF {pdf_number} download failed, skipping")
                    all_results.append({
                        "pdf_number": pdf_number,
                        "status": "failed",
                        "error": "Download failed"
                    })
                    continue

                print(f"\n[Archive] Processing PDF {pdf_number}/{len(pdf_urls)}")

                result = self.pdf_processor.process_pdf_to_ocr(
                    url, pdf_path, pdf_number, book_name, archive_id,
                    pdf_batch_size=archive_pdf_batch_size,
                    image_zoom=archive_image_zoom,
                    image_dpi=archive_image_dpi,
                    ocr_oem=archive_ocr_oem,
                    ocr_psm=archive_ocr_psm,
                    ocr_lang=archive_ocr_lang,
                    ocr_workers=archive_ocr_workers
                )

                all_results.append(result)

                if result["status"] == "completed":
                    success_count += 1
                    total_chars += result.get("total_text_chars", 0)
                    total_vectors += result.get("pinecone_vectors", 0)

                try:
                    os.unlink(pdf_path)
                except:
                    pass

                self.mongo.update_status(
                    archive_id,
                    "processing",
                    current_pdf=pdf_number,
                    total_pdfs=len(pdf_urls),
                    completed_pdfs=success_count,
                    total_chars_processed=total_chars,
                    total_vectors_uploaded=total_vectors,
                    results=all_results
                )

                if idx < len(pdf_urls) - 1:
                    time.sleep(PDF_SLEEP_BETWEEN)

            final_status = "completed" if success_count == len(pdf_urls) else "partial"

            self.mongo.update_status(
                archive_id,
                final_status,
                completed_at=datetime.utcnow(),
                total_pdfs=len(pdf_urls),
                completed_pdfs=success_count,
                total_chars_processed=total_chars,
                total_vectors_uploaded=total_vectors,
                results=all_results
            )

            print(f"\n[Archive] Complete: {success_count}/{len(pdf_urls)} PDFs successful")
            print(f"[Archive] Total: {total_chars} characters, {total_vectors} Pinecone vectors")

            mem_info = get_memory_info()
            print(f"[Archive] Memory: {mem_info.get('process_mb', 0)} MB used")

            return success_count > 0

        except Exception as e:
            print(f"[Archive] Error: {e}")
            self.mongo.update_status(archive_id, "failed", error=str(e))
            self.mongo.increment_retry(archive_id)
            return False

    def _generate_pdf_urls_optimized(self, archive_url: str) -> List[str]:
        """Generate PDF URLs efficiently"""
        pdf_urls = []

        if '/details/' in archive_url:
            item_id = archive_url.split('/details/')[-1].split('/')[0]
        elif '/download/' in archive_url:
            item_id = archive_url.split('/download/')[-1].split('/')[0]
        else:
            return []

        print(f"[Archive] Item ID: {item_id}")

        for i in range(1, 51):
            pdf_url = f"https://archive.org/download/{item_id}/{i}.pdf"
            pdf_urls.append(pdf_url)

        return pdf_urls[:50]

    def run(self):
        """Main processing loop"""
        print(f"[Main] Starting Tafsir OCR Processor for GitHub Actions")
        print(f"[Main] MongoDB: {MONGODB_DB}.{MONGODB_COLLECTION}")
        print(f"[Main] Pinecone Index: {PINECONE_INDEX_NAME}")
        print(f"[Main] Default Max PDFs per run: {MAX_PDFS_PER_RUN}")
        print(f"[Main] Default Image Quality: {IMAGE_ZOOM}x zoom @ {IMAGE_DPI} DPI")
        print(f"[Main] Default OCR: OEM={OCR_OEM}, PSM={OCR_PSM}, Lang={OCR_LANG}")

        try:
            pending = self.mongo.get_pending_archives()

            if not pending:
                print("[Main] No pending archives")
                return

            print(f"[Main] Found {len(pending)} pending archives")

            for idx, archive in enumerate(pending):
                print(f"\n[Main] Archive {idx+1}/{len(pending)}")

                success = self.process_archive_optimized(archive)

                if idx < len(pending) - 1:
                    time.sleep(2)

            print("\n[Main] All processing complete!")

        except Exception as e:
            print(f"[Main] Fatal error: {e}")
            raise

        finally:
            self.mongo.close()
            self.pinecone.close()

# ============ Entry Point ============

def main():
    """Main entry point"""

    required_vars = ["MONGODB_URI", "PINECONE_API_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        print(f"[Error] Missing: {', '.join(missing)}")
        exit(1)

    print(f"[GitHub Actions] Runner: {os.environ.get('RUNNER_OS', 'Unknown')}")
    print(f"[GitHub Actions] Workspace: {os.environ.get('GITHUB_WORKSPACE', 'Local')}")

    processor = TafsirProcessor()
    processor.run()

if __name__ == "__main__":
    main()