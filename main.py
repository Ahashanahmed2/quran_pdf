#!/usr/bin/env python3
"""
Internet Archive PDF Processor - GitHub Actions Optimized
Full utilization of 7GB RAM for maximum throughput
Supports per-archive processing settings from MongoDB
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
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import fitz  # PyMuPDF
from pymongo import MongoClient
from huggingface_hub import HfApi, CommitOperationAdd

# ============ GitHub Actions Optimized Configuration ============

# GitHub Actions provides 7GB RAM - Let's use it efficiently!
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "tafsir_db")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "archive_links")

# Optimized settings for GitHub Actions (7GB RAM) - Default fallback values
MAX_PDFS_PER_RUN = int(os.environ.get("MAX_PDFS_PER_RUN", "50"))
PDF_BATCH_SIZE = int(os.environ.get("PDF_BATCH_SIZE", "100"))
MAX_FILES_PER_COMMIT = int(os.environ.get("MAX_FILES_PER_COMMIT", "100"))
PDF_SLEEP_BETWEEN = int(os.environ.get("PDF_SLEEP_BETWEEN", "1"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Parallel processing settings
MAX_PARALLEL_PDFS = int(os.environ.get("MAX_PARALLEL_PDFS", "3"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

# High-quality image settings (RAM allows it)
IMAGE_ZOOM = float(os.environ.get("IMAGE_ZOOM", "4.0"))
IMAGE_DPI = int(os.environ.get("IMAGE_DPI", "300"))

# Temp Directory
TEMP_DIR = Path(os.environ.get("TEMP_DIR", "/tmp/tafsir_processor"))
TEMP_DIR.mkdir(exist_ok=True, parents=True)

ssl_context = ssl.create_default_context()

# ============ Memory Monitor (GitHub Actions) ============

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

# ============ MongoDB Helper (Optimized) ============

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
        
        # Ensure each archive has processing_settings (add defaults if missing)
        for archive in archives:
            if "processing_settings" not in archive:
                archive["processing_settings"] = {
                    "pdf_batch_size": PDF_BATCH_SIZE,
                    "max_files_per_commit": MAX_FILES_PER_COMMIT,
                    "max_pdfs_per_run": MAX_PDFS_PER_RUN,
                    "image_zoom": IMAGE_ZOOM,
                    "image_dpi": IMAGE_DPI,
                    "max_parallel_pdfs": MAX_PARALLEL_PDFS,
                    "max_workers": MAX_WORKERS
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

    def save_batch_progress(self, archive_id: str, progress_data: Dict):
        """Save progress for multiple PDFs at once"""
        update_data = {}
        for pdf_num, pages in progress_data.items():
            update_data[f"progress.{pdf_num}"] = pages

        if update_data:
            self.collection.update_one(
                {"_id": archive_id},
                {"$set": update_data}
            )

    def close(self):
        self.client.close()

# ============ HuggingFace Helper (Optimized) ============

class HFHelper:
    def __init__(self):
        self.api = HfApi(token=HF_TOKEN)
        self.upload_cache = {}  # Cache uploaded pages to reduce API calls

    def get_uploaded_pages_batch(self, folder_path: str, pdf_numbers: List[str]) -> Dict[str, set]:
        """Batch check uploaded pages for multiple PDFs"""
        result = {}

        try:
            files = self.api.list_files_info(
                repo_id=HF_DATASET,
                repo_type="dataset",
                path=folder_path
            )

            # Group files by PDF number
            for file in files:
                parts = file.rfilename.split('/')
                if len(parts) >= 2:
                    pdf_num = parts[-2] if parts[-2].isdigit() else None
                    if pdf_num and pdf_num in pdf_numbers:
                        if pdf_num not in result:
                            result[pdf_num] = set()

                        filename = parts[-1]
                        if filename.startswith("page_") and filename.endswith(".png"):
                            try:
                                page_num = int(filename.replace("page_", "").replace(".png", ""))
                                result[pdf_num].add(page_num)
                            except:
                                pass

            # Add empty sets for PDFs with no uploads
            for pdf_num in pdf_numbers:
                if pdf_num not in result:
                    result[pdf_num] = set()

        except Exception as e:
            print(f"[HF] Batch check error: {e}")
            for pdf_num in pdf_numbers:
                result[pdf_num] = set()

        return result

    def upload_batch(self, local_folder: str, hf_path: str, batch_name: str) -> bool:
        """Upload with default max files per commit"""
        return self.upload_batch_with_limit(local_folder, hf_path, batch_name, MAX_FILES_PER_COMMIT)

    def upload_batch_with_limit(self, local_folder: str, hf_path: str, batch_name: str, max_files_per_commit: int) -> bool:
        """Upload with custom max files per commit limit"""
        try:
            png_files = sorted(Path(local_folder).glob("*.png"))

            if not png_files:
                return True

            print(f"[HF] Uploading {len(png_files)} files (Max per commit: {max_files_per_commit})")

            # GitHub Actions can handle larger commits
            for i in range(0, len(png_files), max_files_per_commit):
                chunk_files = png_files[i:i+max_files_per_commit]
                operations = []

                for png_file in chunk_files:
                    remote_path = f"{hf_path}/{png_file.name}"
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=remote_path,
                            path_or_fileobj=str(png_file)
                        )
                    )

                chunk_num = i // max_files_per_commit + 1
                total_chunks = (len(png_files) + max_files_per_commit - 1) // max_files_per_commit

                self.api.create_commit(
                    repo_id=HF_DATASET,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"📚 {batch_name} - part{chunk_num} ({len(operations)} pages)"
                )

                print(f"[HF] ✅ Uploaded chunk {chunk_num}/{total_chunks}")

            return True

        except Exception as e:
            print(f"[HF] ❌ Upload failed: {e}")
            return False

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

# ============ Optimized PDF Processor ============

class PDFProcessor:
    def __init__(self, hf_helper: HFHelper):
        self.hf = hf_helper

    def process_pdf_optimized(self, pdf_url: str, pdf_path: str, pdf_number: int, 
                              folder_name: str, archive_id: str, 
                              mongo_helper: MongoDBHelper,
                              pdf_batch_size: int = PDF_BATCH_SIZE,
                              max_files_per_commit: int = MAX_FILES_PER_COMMIT,
                              image_zoom: float = IMAGE_ZOOM,
                              image_dpi: int = IMAGE_DPI) -> Dict:
        """Process PDF with archive-specific optimized settings"""

        result = {
            "pdf_number": pdf_number,
            "status": "pending",
            "pages_processed": 0,
            "pages_uploaded": 0
        }

        clean_folder = self._clean_folder_name(folder_name)
        hf_path = f"{clean_folder}/{pdf_number}"

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            print(f"[PDF {pdf_number}] {total_pages} pages total (Batch={pdf_batch_size}, Zoom={image_zoom}, DPI={image_dpi})")

            # Process in batches using archive-specific batch size
            pages_processed = 0

            for batch_start in range(0, total_pages, pdf_batch_size):
                batch_end = min(batch_start + pdf_batch_size, total_pages)
                batch_number = (batch_start // pdf_batch_size) + 1

                batch_folder = Path(tempfile.mkdtemp(
                    prefix=f"pdf{pdf_number}_batch{batch_number}_", 
                    dir=TEMP_DIR
                ))

                try:
                    # Open PDF for this batch
                    batch_doc = fitz.open(pdf_path)
                    pages_in_batch = 0

                    # Render all pages in batch at once
                    for page_num in range(batch_start, batch_end):
                        page = batch_doc.load_page(page_num)

                        # Archive-specific zoom
                        mat = fitz.Matrix(image_zoom, image_zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)

                        img_name = f"page_{page_num+1:04d}.png"
                        img_path = batch_folder / img_name
                        pix.save(img_path, "png")

                        pages_in_batch += 1
                        pages_processed += 1

                        # Less aggressive cleanup (more RAM available)
                        pix = None
                        page = None

                    batch_doc.close()

                    # Upload batch using archive-specific max files per commit
                    if pages_in_batch > 0:
                        upload_success = self.hf.upload_batch_with_limit(
                            str(batch_folder),
                            hf_path,
                            f"PDF{pdf_number}_batch{batch_number}",
                            max_files_per_commit=max_files_per_commit
                        )

                        if upload_success:
                            print(f"[PDF {pdf_number}] ✅ Batch {batch_number}: {pages_in_batch} pages")
                        else:
                            raise Exception(f"Batch {batch_number} upload failed")

                finally:
                    # Cleanup
                    try:
                        if batch_folder.exists():
                            import shutil
                            shutil.rmtree(batch_folder, ignore_errors=True)
                    except:
                        pass

                # Minimal pause between batches
                if batch_end < total_pages:
                    time.sleep(0.5)

            result["status"] = "completed"
            result["pages_processed"] = pages_processed
            result["pages_uploaded"] = pages_processed

        except Exception as e:
            print(f"[PDF {pdf_number}] ❌ Error: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    @staticmethod
    def _clean_folder_name(name: str) -> str:
        """Clean folder name"""
        import re
        return re.sub(r'[^\w\-_]', '_', name.replace(' ', '_'))

# ============ Main Processor (Optimized) ============

class TafsirProcessor:
    def __init__(self):
        self.mongo = MongoDBHelper()
        self.hf = HFHelper()
        self.pdf_processor = PDFProcessor(self.hf)

        # Print memory info at start
        mem_info = get_memory_info()
        print(f"[System] RAM: {mem_info.get('total_gb', 'N/A')} GB total, "
              f"{mem_info.get('available_gb', 'N/A')} GB available")

    def process_archive_optimized(self, archive_item: Dict) -> bool:
        """Process archive with parallel downloads and optimized settings"""

        archive_id = archive_item["_id"]
        archive_url = archive_item["url"]
        book_name = archive_item.get("book_name", f"tafsir_{archive_id}")
        
        # Get archive-specific settings (fallback to global defaults)
        settings = archive_item.get("processing_settings", {})
        archive_pdf_batch_size = settings.get("pdf_batch_size", PDF_BATCH_SIZE)
        archive_max_files_per_commit = settings.get("max_files_per_commit", MAX_FILES_PER_COMMIT)
        archive_max_pdfs_per_run = settings.get("max_pdfs_per_run", MAX_PDFS_PER_RUN)
        archive_image_zoom = settings.get("image_zoom", IMAGE_ZOOM)
        archive_image_dpi = settings.get("image_dpi", IMAGE_DPI)
        archive_max_parallel_pdfs = settings.get("max_parallel_pdfs", MAX_PARALLEL_PDFS)
        archive_max_workers = settings.get("max_workers", MAX_WORKERS)

        print(f"\n{'='*70}")
        print(f"[Archive] {book_name}")
        print(f"[Archive] URL: {archive_url}")
        print(f"[Archive] Settings: Batch={archive_pdf_batch_size}, MaxFiles={archive_max_files_per_commit}, MaxPDFs={archive_max_pdfs_per_run}, Zoom={archive_image_zoom}, DPI={archive_image_dpi}, Workers={archive_max_workers}")
        print(f"{'='*70}\n")

        try:
            self.mongo.update_status(archive_id, "processing", started_at=datetime.utcnow())

            # Generate PDF URLs
            pdf_urls = self._generate_pdf_urls_optimized(archive_url)

            if not pdf_urls:
                print(f"[Archive] No PDFs found")
                self.mongo.update_status(archive_id, "failed", error="No PDFs found")
                return False

            # Apply Max PDFs per run limit (archive-specific)
            if len(pdf_urls) > archive_max_pdfs_per_run:
                print(f"[Archive] Limiting PDFs from {len(pdf_urls)} to {archive_max_pdfs_per_run}")
                pdf_urls = pdf_urls[:archive_max_pdfs_per_run]

            print(f"[Archive] Found {len(pdf_urls)} PDFs to process")

            # Download PDFs in parallel using archive-specific workers
            print(f"[Archive] Downloading {len(pdf_urls)} PDFs in parallel (workers={archive_max_workers})...")
            download_results = PDFDownloader.download_parallel(pdf_urls, max_workers=archive_max_workers)

            # Process downloaded PDFs
            all_results = []
            success_count = 0

            for idx, url in enumerate(pdf_urls):
                pdf_number = idx + 1
                pdf_path = download_results.get(url)

                if not pdf_path:
                    print(f"[Archive] ⚠️ PDF {pdf_number} download failed, skipping")
                    all_results.append({
                        "pdf_number": pdf_number,
                        "status": "failed",
                        "error": "Download failed"
                    })
                    continue

                print(f"\n[Archive] Processing PDF {pdf_number}/{len(pdf_urls)}")

                result = self.pdf_processor.process_pdf_optimized(
                    url, pdf_path, pdf_number, book_name, archive_id, self.mongo,
                    pdf_batch_size=archive_pdf_batch_size,
                    max_files_per_commit=archive_max_files_per_commit,
                    image_zoom=archive_image_zoom,
                    image_dpi=archive_image_dpi
                )

                all_results.append(result)

                if result["status"] == "completed":
                    success_count += 1

                # Cleanup PDF file
                try:
                    os.unlink(pdf_path)
                except:
                    pass

                # Update MongoDB
                self.mongo.update_status(
                    archive_id,
                    "processing",
                    current_pdf=pdf_number,
                    total_pdfs=len(pdf_urls),
                    completed_pdfs=success_count,
                    results=all_results
                )
                
                # Pause between PDFs
                if idx < len(pdf_urls) - 1:
                    time.sleep(PDF_SLEEP_BETWEEN)

            # Final status
            final_status = "completed" if success_count == len(pdf_urls) else "partial"

            self.mongo.update_status(
                archive_id,
                final_status,
                completed_at=datetime.utcnow(),
                total_pdfs=len(pdf_urls),
                completed_pdfs=success_count,
                results=all_results
            )

            print(f"\n[Archive] ✅ Complete: {success_count}/{len(pdf_urls)} PDFs successful")

            # Print memory info
            mem_info = get_memory_info()
            print(f"[Archive] Memory: {mem_info.get('process_mb', 0)} MB used")

            return success_count > 0

        except Exception as e:
            print(f"[Archive] ❌ Error: {e}")
            self.mongo.update_status(archive_id, "failed", error=str(e))
            self.mongo.increment_retry(archive_id)
            return False

    def _generate_pdf_urls_optimized(self, archive_url: str) -> List[str]:
        """Generate PDF URLs efficiently"""
        pdf_urls = []

        # Extract item ID
        if '/details/' in archive_url:
            item_id = archive_url.split('/details/')[-1].split('/')[0]
        elif '/download/' in archive_url:
            item_id = archive_url.split('/download/')[-1].split('/')[0]
        else:
            return []

        print(f"[Archive] Item ID: {item_id}")

        # Try up to 50 PDFs (more aggressive)
        for i in range(1, 51):
            pdf_url = f"https://archive.org/download/{item_id}/{i}.pdf"
            pdf_urls.append(pdf_url)  # Add all, filter later

        return pdf_urls[:50]  # Limit to 50 max

    def run(self):
        """Main processing loop"""
        print(f"[Main] 🚀 Starting Optimized Tafsir Processor for GitHub Actions")
        print(f"[Main] MongoDB: {MONGODB_DB}.{MONGODB_COLLECTION}")
        print(f"[Main] HF Dataset: {HF_DATASET}")
        print(f"[Main] Default Max PDFs per run: {MAX_PDFS_PER_RUN}")
        print(f"[Main] Default PDF Batch Size: {PDF_BATCH_SIZE}")
        print(f"[Main] Default Image Quality: {IMAGE_ZOOM}x zoom @ {IMAGE_DPI} DPI")

        try:
            pending = self.mongo.get_pending_archives()

            if not pending:
                print("[Main] No pending archives")
                return

            print(f"[Main] Found {len(pending)} pending archives")

            for idx, archive in enumerate(pending):
                print(f"\n[Main] 📦 Archive {idx+1}/{len(pending)}")

                success = self.process_archive_optimized(archive)

                if idx < len(pending) - 1:
                    time.sleep(2)

            print("\n[Main] ✅ All processing complete!")

        except Exception as e:
            print(f"[Main] ❌ Fatal error: {e}")
            raise

        finally:
            self.mongo.close()

# ============ Entry Point ============

def main():
    """Main entry point"""

    # Validate environment
    required_vars = ["HF_TOKEN", "HF_DATASET", "MONGODB_URI"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        print(f"[Error] Missing: {', '.join(missing)}")
        exit(1)

    # Show GitHub Actions environment info
    print(f"[GitHub Actions] Runner: {os.environ.get('RUNNER_OS', 'Unknown')}")
    print(f"[GitHub Actions] Workspace: {os.environ.get('GITHUB_WORKSPACE', 'Local')}")

    # Run processor
    processor = TafsirProcessor()
    processor.run()

if __name__ == "__main__":
    main()