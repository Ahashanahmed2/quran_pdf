#!/usr/bin/env python3
"""
Render.com Web UI for Tafsir PDF Processor Configuration
Complete management interface for MongoDB and Pinecone
Mobile Responsive Design with OCR Language Selection
"""

import os
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError, DuplicateKeyError
from huggingface_hub import HfApi
import uvicorn

# ============ Configuration ============

# Default values (will be overridden by MongoDB config)
DEFAULT_CONFIG = {
    "hf_token": "",
    "hf_dataset": "",
    "mongodb_uri": "",
    "mongodb_db": "tafsir_db",
    "mongodb_collection": "archive_links",
    "pinecone_api_key": "",
    "pinecone_index_name": "tafsir-ocr",
    "priority_default": 5
}

# Local config file for Render.com (fallback)
CONFIG_FILE = Path("/data/config.json") if os.path.exists("/data") else Path("config.json")

# ============ Models ============

class SystemConfig(BaseModel):
    """Complete system configuration"""
    hf_token: str = Field("", description="HuggingFace API Token")
    hf_dataset: str = Field("", description="HuggingFace Dataset (username/dataset_name)")
    mongodb_uri: str = Field(..., description="MongoDB Connection URI")
    mongodb_db: str = Field("tafsir_db", description="MongoDB Database Name")
    mongodb_collection: str = Field("archive_links", description="MongoDB Collection Name")
    pinecone_api_key: str = Field("", description="Pinecone API Key")
    pinecone_index_name: str = Field("tafsir-ocr", description="Pinecone Index Name")
    priority_default: int = Field(5, ge=1, le=10, description="Default priority for new archives")

class ArchiveItem(BaseModel):
    """Archive item to process"""
    book_name: str = Field(..., description="Book name in Bengali/English")
    archive_url: str = Field(..., description="Internet Archive URL")
    priority: int = Field(5, ge=1, le=10, description="Processing priority (1-10)")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

    # Processing settings for this specific archive
    pdf_batch_size: int = Field(50, ge=10, le=200)
    max_files_per_commit: int = Field(50, ge=10, le=100)
    max_pdfs_per_run: int = Field(20, ge=1, le=100)
    image_zoom: float = Field(3.0, ge=1.0, le=5.0)
    image_dpi: int = Field(200, ge=72, le=400)
    max_parallel_pdfs: int = Field(2, ge=1, le=5)
    max_workers: int = Field(2, ge=1, le=5)
    
    # OCR Settings
    ocr_oem: int = Field(3, ge=0, le=3)
    ocr_psm: int = Field(3, ge=0, le=13)
    ocr_lang: str = Field("ben", description="OCR Languages (e.g., ben+ara+eng)")
    ocr_workers: int = Field(2, ge=1, le=4)

class ArchiveUpdateModel(BaseModel):
    """Update archive item"""
    book_name: Optional[str] = None
    archive_url: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    status: Optional[str] = None
    metadata: Optional[Dict] = None
    pdf_batch_size: Optional[int] = Field(None, ge=10, le=200)
    max_files_per_commit: Optional[int] = Field(None, ge=10, le=100)
    max_pdfs_per_run: Optional[int] = Field(None, ge=1, le=100)
    image_zoom: Optional[float] = Field(None, ge=1.0, le=5.0)
    image_dpi: Optional[int] = Field(None, ge=72, le=400)
    max_parallel_pdfs: Optional[int] = Field(None, ge=1, le=5)
    max_workers: Optional[int] = Field(None, ge=1, le=5)
    ocr_oem: Optional[int] = Field(None, ge=0, le=3)
    ocr_psm: Optional[int] = Field(None, ge=0, le=13)
    ocr_lang: Optional[str] = None
    ocr_workers: Optional[int] = Field(None, ge=1, le=4)
    retry_count: Optional[int] = Field(None, ge=0, le=10)

class BulkArchiveInput(BaseModel):
    """Bulk archive input"""
    items: List[ArchiveItem]

# ============ Config Manager ============

class ConfigManager:
    """Manage system configuration in MongoDB"""

    def __init__(self):
        self.config_collection = None
        self.client = None
        self.db = None
        self.is_connected = False
        self.archive_collection = None

    def initialize(self, mongodb_uri: str = None, db_name: str = "tafsir_config"):
        """Initialize MongoDB connection"""
        if not mongodb_uri:
            print("[ConfigManager] No MongoDB URI provided")
            return False

        try:
            print(f"[ConfigManager] Connecting to MongoDB...")
            self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
            self.client.admin.command('ping')

            self.db = self.client[db_name]
            self.config_collection = self.db["system_config"]

            # Archive collection initialize
            data_db_name = DEFAULT_CONFIG.get("mongodb_db", "tafsir_db")
            data_collection_name = DEFAULT_CONFIG.get("mongodb_collection", "archive_links")
            data_db = self.client[data_db_name]
            self.archive_collection = data_db[data_collection_name]

            self.is_connected = True
            print(f"[ConfigManager] Connected to MongoDB: {db_name}")
            return True

        except Exception as e:
            print(f"[ConfigManager] MongoDB connection failed: {e}")
            self.is_connected = False
            return False

    def get_config(self) -> Dict:
        """Get current configuration"""
        if self.is_connected and self.config_collection is not None:
            try:
                config = self.config_collection.find_one({"_id": "current"})
                if config:
                    config.pop("_id", None)
                    print("[ConfigManager] Loaded config from MongoDB")
                    return config
            except Exception as e:
                print(f"[ConfigManager] Failed to load from MongoDB: {e}")

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    print("[ConfigManager] Loaded config from file")
                    return config
            except Exception as e:
                print(f"[ConfigManager] Failed to load from file: {e}")

        print("[ConfigManager] Using default config")
        return DEFAULT_CONFIG.copy()

    def save_config(self, config: Dict) -> bool:
        """Save configuration"""
        config["updated_at"] = datetime.utcnow().isoformat()

        if self.is_connected and self.config_collection is not None:
            try:
                result = self.config_collection.update_one(
                    {"_id": "current"},
                    {"$set": config},
                    upsert=True
                )
                print(f"[ConfigManager] Saved to MongoDB. Modified: {result.modified_count}, Upserted: {result.upserted_id}")

                saved = self.config_collection.find_one({"_id": "current"})
                if saved:
                    print(f"[ConfigManager] Verified - Config exists in MongoDB")

                    if config.get("mongodb_uri") and config.get("mongodb_db") and config.get("mongodb_collection"):
                        try:
                            data_db = self.client[config["mongodb_db"]]
                            self.archive_collection = data_db[config["mongodb_collection"]]
                            print(f"[ConfigManager] Archive collection initialized: {config['mongodb_db']}.{config['mongodb_collection']}")
                        except Exception as e:
                            print(f"[ConfigManager] Failed to initialize archive collection: {e}")

                    return True

            except Exception as e:
                print(f"[ConfigManager] Failed to save to MongoDB: {e}")
        else:
            print("[ConfigManager] Not connected to MongoDB, saving to file only")

        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            print(f"[ConfigManager] Saved to file: {CONFIG_FILE}")
            return True
        except Exception as e:
            print(f"[ConfigManager] Failed to save to file: {e}")
            return False

    def test_mongodb_connection(self, uri: str) -> tuple:
        """Test MongoDB connection"""
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            server_info = client.server_info()
            client.close()
            return True, f"Connected to MongoDB {server_info.get('version', 'Unknown')}"
        except Exception as e:
            return False, str(e)

    def test_hf_connection(self, token: str) -> tuple:
        """Test HuggingFace connection"""
        try:
            api = HfApi(token=token)
            user = api.whoami()
            return True, f"Connected as {user.get('name', 'Unknown')}"
        except Exception as e:
            return False, str(e)

    def get_archives(self, limit: int = 100) -> List[Dict]:
        """Get all archive items"""
        if not self.is_connected or self.archive_collection is None:
            print("[ConfigManager] Not connected to archive collection")
            return []

        try:
            archives = list(self.archive_collection.find().sort("created_at", -1).limit(limit))
            for archive in archives:
                archive["_id"] = str(archive["_id"])
                if "created_at" in archive:
                    archive["created_at"] = archive["created_at"].isoformat()
                if "updated_at" in archive:
                    archive["updated_at"] = archive["updated_at"].isoformat()
            return archives
        except Exception as e:
            print(f"[ConfigManager] Failed to fetch archives: {e}")
            return []

    def add_archive(self, archive_data: Dict) -> tuple:
        """Add a new archive item"""
        if not self.is_connected or self.archive_collection is None:
            return False, "MongoDB not connected"

        try:
            import re
            doc_id = re.sub(r'[^\w\-_]', '_', archive_data.get("book_name", "").lower().replace(' ', '_'))

            # Check duplicate by ID
            existing = self.archive_collection.find_one({"_id": doc_id})
            if existing:
                return False, f"Archive already exists with Book Name: '{archive_data.get('book_name')}'"

            # Check duplicate by URL
            existing_url = self.archive_collection.find_one({"url": archive_data.get("url")})
            if existing_url:
                return False, f"URL already exists under: '{existing_url.get('book_name')}'"

            document = {
                "_id": doc_id,
                **archive_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            self.archive_collection.insert_one(document)
            return True, doc_id

        except DuplicateKeyError:
            return False, "Archive URL already exists"
        except Exception as e:
            return False, str(e)

    def update_archive(self, archive_id: str, update_data: Dict) -> tuple:
        """Update an archive item"""
        if not self.is_connected or self.archive_collection is None:
            return False, "MongoDB not connected"

        try:
            existing = self.archive_collection.find_one({"_id": archive_id})
            if not existing:
                return False, "Archive not found"

            # Check URL duplicate if changing
            if "url" in update_data and update_data["url"] != existing.get("url"):
                url_exists = self.archive_collection.find_one({"url": update_data["url"], "_id": {"$ne": archive_id}})
                if url_exists:
                    return False, f"URL already exists under: '{url_exists.get('book_name')}'"

            update_data["updated_at"] = datetime.utcnow()

            # Handle nested processing_settings
            final_update = {}
            for key, value in update_data.items():
                if key.startswith("processing_settings."):
                    final_update[key] = value
                else:
                    final_update[key] = value

            self.archive_collection.update_one({"_id": archive_id}, {"$set": final_update})
            return True, "Archive updated successfully"

        except Exception as e:
            return False, str(e)

    def delete_archive(self, archive_id: str) -> tuple:
        """Delete an archive item"""
        if not self.is_connected or self.archive_collection is None:
            return False, "MongoDB not connected"

        try:
            result = self.archive_collection.delete_one({"_id": archive_id})
            if result.deleted_count > 0:
                return True, "Archive deleted successfully"
            else:
                return False, "Archive not found"
        except Exception as e:
            return False, str(e)

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        if not self.is_connected or self.archive_collection is None:
            return {
                "active_tasks": 0,
                "total_completed": 0,
                "total_pending": 0,
                "total_failed": 0
            }

        try:
            return {
                "active_tasks": self.archive_collection.count_documents({"status": "processing"}),
                "total_completed": self.archive_collection.count_documents({"status": "completed"}),
                "total_pending": self.archive_collection.count_documents({"status": "pending"}),
                "total_failed": self.archive_collection.count_documents({"status": "failed"})
            }
        except Exception as e:
            print(f"[ConfigManager] Failed to get statistics: {e}")
            return {
                "active_tasks": 0,
                "total_completed": 0,
                "total_pending": 0,
                "total_failed": 0
            }

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.config_collection = None
            self.archive_collection = None
            self.is_connected = False

# ============ FastAPI App ============

app = FastAPI(title="Tafsir PDF Processor Config", version="3.0.0")

# Initialize config manager
config_manager = ConfigManager()

# ============ HTML Templates ============

HTML_HEADER = """
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
    <title>তাফসীর PDF প্রসেসর - কনফিগারেশন</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 15px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            color: #1a5f7a;
            font-size: 24px;
        }
        .nav-tabs {
            display: flex;
            gap: 10px;
        }
        .nav-tab {
            padding: 12px 24px;
            background: #f0f0f0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .nav-tab:hover {
            background: #e0e0e0;
        }
        .nav-tab.active {
            background: #1a5f7a;
            color: white;
        }
        .tab-content {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 16px;
        }
        input, select, textarea {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s, box-shadow 0.3s;
            background: white;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #1a5f7a;
            box-shadow: 0 0 0 4px rgba(26, 95, 122, 0.1);
        }
        input[type="number"] {
            -moz-appearance: textfield;
        }
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            white-space: nowrap;
        }
        .btn-primary {
            background: linear-gradient(135deg, #1a5f7a 0%, #0d3b4c 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(26, 95, 122, 0.4);
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        .btn-warning {
            background: #ffc107;
            color: #333;
        }
        .btn-group {
            display: flex;
            gap: 15px;
            margin-top: 25px;
            flex-wrap: wrap;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
        }
        .status-pending { background: #fff3cd; color: #856404; }
        .status-processing { background: #cce5ff; color: #004085; }
        .status-completed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        
        .table-container {
            overflow-x: auto;
            margin-top: 30px;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
            font-size: 14px;
        }
        td {
            color: #555;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .alert {
            padding: 18px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 15px;
        }
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .alert-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        h2 {
            font-size: 24px;
            margin-bottom: 15px;
            color: #333;
        }
        h3 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #444;
        }
        h4 {
            font-size: 16px;
            margin: 20px 0 15px 0;
            color: #555;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        .card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        small {
            display: block;
            margin-top: 5px;
            color: #666;
            font-size: 13px;
        }
        .settings-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #e0e0e0;
        }
        .edit-form-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .edit-form {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 900px;
            max-height: 90vh;
            overflow-y: auto;
            width: 90%;
        }
        
        .mobile-archive-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #1a5f7a;
        }
        
        @media screen and (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .header {
                flex-direction: column;
                gap: 15px;
                padding: 15px 20px;
            }
            
            .header h1 {
                font-size: 20px;
                text-align: center;
            }
            
            .nav-tabs {
                flex-wrap: wrap;
                justify-content: center;
                width: 100%;
            }
            
            .nav-tab {
                padding: 10px 16px;
                font-size: 14px;
                flex: 1 0 auto;
            }
            
            .tab-content {
                padding: 20px 15px;
            }
            
            .form-grid {
                grid-template-columns: 1fr !important;
                gap: 15px;
            }
            
            .form-group.full-width {
                grid-column: span 1;
            }
            
            #add-archive-form > div:first-of-type {
                grid-template-columns: 1fr !important;
                gap: 15px;
            }
            
            .settings-section > div {
                grid-template-columns: 1fr !important;
                gap: 15px;
            }
            
            .btn {
                padding: 12px 16px;
                font-size: 14px;
                width: 100%;
            }
            
            .btn-group {
                flex-direction: column;
                gap: 10px;
            }
            
            .card {
                padding: 15px;
            }
            
            .edit-form {
                padding: 20px;
                width: 95%;
            }
            
            #monitor-tab > div:first-of-type {
                grid-template-columns: 1fr !important;
                gap: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
"""

HTML_FOOTER = """
    </div>
    
    <!-- Edit Form Modal -->
    <div id="edit-modal" class="edit-form-container" onclick="if(event.target===this)closeEditModal()">
        <div class="edit-form">
            <h2>✏️ আর্কাইভ সম্পাদনা</h2>
            <form id="edit-archive-form" onsubmit="updateArchive(event)">
                <input type="hidden" id="edit_archive_id">
                
                <h4>📋 সাধারণ তথ্য</h4>
                <div style="display: grid; gap: 15px;">
                    <div>
                        <label for="edit_book_name">📚 বইয়ের নাম *</label>
                        <input type="text" id="edit_book_name" required style="padding: 14px;">
                    </div>
                    <div>
                        <label for="edit_archive_url">🔗 আর্কাইভ URL *</label>
                        <input type="text" id="edit_archive_url" required style="padding: 14px;">
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <label for="edit_priority">⚡ অগ্রাধিকার (1-10)</label>
                            <input type="number" id="edit_priority" min="1" max="10" style="padding: 14px;">
                        </div>
                        <div>
                            <label for="edit_status">📊 অবস্থা</label>
                            <select id="edit_status" style="padding: 14px;">
                                <option value="pending">অপেক্ষমান</option>
                                <option value="processing">প্রক্রিয়াধীন</option>
                                <option value="completed">সম্পন্ন</option>
                                <option value="failed">ব্যর্থ</option>
                                <option value="partial">আংশিক</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <h4>⚙️ প্রসেসিং সেটিংস</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div>
                        <label for="edit_pdf_batch_size">📦 PDF ব্যাচ সাইজ</label>
                        <input type="number" id="edit_pdf_batch_size" min="10" max="200" style="padding: 12px;">
                    </div>
                    <div>
                        <label for="edit_max_files_per_commit">📤 প্রতি কমিটে সর্বোচ্চ ফাইল</label>
                        <input type="number" id="edit_max_files_per_commit" min="10" max="100" style="padding: 12px;">
                    </div>
                    <div>
                        <label for="edit_max_pdfs_per_run">📚 প্রতি রানে সর্বোচ্চ PDF</label>
                        <input type="number" id="edit_max_pdfs_per_run" min="1" max="100" style="padding: 12px;">
                    </div>
                    <div>
                        <label for="edit_image_zoom">🔍 ইমেজ জুম</label>
                        <input type="number" id="edit_image_zoom" min="1.0" max="5.0" step="0.5" style="padding: 12px;">
                    </div>
                    <div>
                        <label for="edit_image_dpi">🖼️ ইমেজ DPI</label>
                        <input type="number" id="edit_image_dpi" min="72" max="400" style="padding: 12px;">
                    </div>
                    <div>
                        <label for="edit_max_parallel_pdfs">⚡ প্যারালাল PDF</label>
                        <input type="number" id="edit_max_parallel_pdfs" min="1" max="5" style="padding: 12px;">
                    </div>
                    <div>
                        <label for="edit_max_workers">🔄 ডাউনলোড ওয়ার্কার</label>
                        <input type="number" id="edit_max_workers" min="1" max="5" style="padding: 12px;">
                    </div>
                </div>
                
                <h4>🔤 OCR সেটিংস</h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div>
                        <label for="edit_ocr_lang_1">ভাষা 1 (প্রাথমিক)</label>
                        <select id="edit_ocr_lang_1" style="padding: 12px;">
                            <option value="ben" selected>🇧🇩 বাংলা</option>
                            <option value="ara">🇸🇦 আরবি</option>
                            <option value="eng">🇬🇧 ইংরেজি</option>
                            <option value="urd">🇵🇰 উর্দু</option>
                            <option value="fas">🇮🇷 ফারসি</option>
                            <option value="hin">🇮🇳 হিন্দি</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_lang_2">ভাষা 2 (ঐচ্ছিক)</label>
                        <select id="edit_ocr_lang_2" style="padding: 12px;">
                            <option value="" selected>-- কোনোটি নয় --</option>
                            <option value="ben">🇧🇩 বাংলা</option>
                            <option value="ara">🇸🇦 আরবি</option>
                            <option value="eng">🇬🇧 ইংরেজি</option>
                            <option value="urd">🇵🇰 উর্দু</option>
                            <option value="fas">🇮🇷 ফারসি</option>
                            <option value="hin">🇮🇳 হিন্দি</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_lang_3">ভাষা 3 (ঐচ্ছিক)</label>
                        <select id="edit_ocr_lang_3" style="padding: 12px;">
                            <option value="" selected>-- কোনোটি নয় --</option>
                            <option value="ben">🇧🇩 বাংলা</option>
                            <option value="ara">🇸🇦 আরবি</option>
                            <option value="eng">🇬🇧 ইংরেজি</option>
                            <option value="urd">🇵🇰 উর্দু</option>
                            <option value="fas">🇮🇷 ফারসি</option>
                            <option value="hin">🇮🇳 হিন্দি</option>
                        </select>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                    <div>
                        <label for="edit_ocr_oem">OCR ইঞ্জিন মোড</label>
                        <select id="edit_ocr_oem" style="padding: 12px;">
                            <option value="3" selected>3 - LSTM নিউরাল (সেরা)</option>
                            <option value="1">1 - শুধু LSTM</option>
                            <option value="2">2 - LSTM + লিগ্যাসি</option>
                            <option value="0">0 - শুধু লিগ্যাসি</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_psm">পৃষ্ঠা সেগমেন্টেশন</label>
                        <select id="edit_ocr_psm" style="padding: 12px;">
                            <option value="3" selected>3 - স্বয়ংক্রিয়</option>
                            <option value="6">6 - সমান টেক্সট ব্লক</option>
                            <option value="1">1 - OSD সহ স্বয়ংক্রিয়</option>
                            <option value="4">4 - একক কলাম</option>
                            <option value="7">7 - একক টেক্সট লাইন</option>
                            <option value="8">8 - একক শব্দ</option>
                            <option value="11">11 - বিক্ষিপ্ত টেক্সট</option>
                            <option value="12">12 - OSD সহ বিক্ষিপ্ত</option>
                            <option value="13">13 - র' লাইন</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_workers">OCR ওয়ার্কার</label>
                        <input type="number" id="edit_ocr_workers" min="1" max="4" value="2" style="padding: 12px;">
                        <small>প্যারালাল OCR থ্রেড</small>
                    </div>
                </div>
                
                <div class="btn-group" style="margin-top: 25px;">
                    <button type="submit" class="btn btn-primary">💾 আপডেট করুন</button>
                    <button type="button" class="btn btn-secondary" onclick="closeEditModal()">❌ বাতিল</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        function isMobile() {
            return window.innerWidth <= 768;
        }
        
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.nav-tab').forEach(btn => {
                btn.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            
            if (tabId === 'monitor-tab') {
                loadMonitorStatus();
            }
        }
        
        async function testMongoDB() {
            const uri = document.getElementById('mongodb_uri').value;
            const resultDiv = document.getElementById('mongodb-test-result');
            resultDiv.innerHTML = 'Testing...';
            try {
                const response = await fetch('/api/test/mongodb', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({uri: uri})
                });
                const data = await response.json();
                if (data.success) {
                    resultDiv.innerHTML = `<div class="alert alert-success">✅ ${data.message}</div>`;
                } else {
                    resultDiv.innerHTML = `<div class="alert alert-error">❌ ${data.message}</div>`;
                }
            } catch (e) {
                resultDiv.innerHTML = `<div class="alert alert-error">❌ Connection failed</div>`;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function formatMobileCard(archive) {
            return `
                <div class="mobile-archive-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                        <strong style="font-size: 16px;">${archive.book_name || 'N/A'}</strong>
                        <span class="status-badge status-${archive.status || 'pending'}">${archive.status || 'pending'}</span>
                    </div>
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px; word-break: break-all;">
                        ${archive.url ? archive.url.substring(0, 50) + '...' : 'N/A'}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 12px; margin-bottom: 10px;">
                        <div><span style="color: #666;">Priority:</span> ${archive.priority || 5}</div>
                        <div><span style="color: #666;">Batch:</span> ${archive.processing_settings?.pdf_batch_size || 50}</div>
                        <div><span style="color: #666;">Progress:</span> ${archive.completed_pdfs || 0}/${archive.total_pdfs || 0}</div>
                    </div>
                    <div style="display: flex; gap: 8px; justify-content: flex-end;">
                        <button class="btn btn-secondary" onclick="editArchive('${archive._id}')" style="padding: 8px 16px; font-size: 12px;">✏️ Edit</button>
                        <button class="btn btn-danger" onclick="deleteArchive('${archive._id}')" style="padding: 8px 16px; font-size: 12px;">🗑️ Delete</button>
                    </div>
                </div>
            `;
        }
        
        function formatTableRow(archive) {
            return `
                <tr>
                    <td><input type="checkbox" value="${archive._id}" style="width: 18px; height: 18px;"></td>
                    <td><strong>${archive.book_name || 'N/A'}</strong></td>
                    <td>${archive.url ? archive.url.substring(0, 40) + '...' : 'N/A'}</td>
                    <td><span class="status-badge status-${archive.status || 'pending'}">${archive.status || 'pending'}</span></td>
                    <td>${archive.priority || 5}</td>
                    <td>${archive.processing_settings?.pdf_batch_size || 50}</td>
                    <td>${archive.completed_pdfs || 0}/${archive.total_pdfs || 0}</td>
                    <td>${archive.updated_at ? new Date(archive.updated_at).toLocaleString('bn-BD') : 'N/A'}</td>
                    <td>
                        <button class="btn btn-secondary" onclick="editArchive('${archive._id}')" style="padding: 6px 12px; font-size: 12px;">✏️</button>
                        <button class="btn btn-danger" onclick="deleteArchive('${archive._id}')" style="padding: 6px 12px; font-size: 12px;">🗑️</button>
                    </td>
                </tr>
            `;
        }
        
        async function loadArchives() {
            try {
                const response = await fetch('/api/archives');
                const archives = await response.json();
                
                if (isMobile()) {
                    const container = document.getElementById('mobile-archives-container');
                    const tableContainer = document.querySelector('.table-container');
                    
                    if (archives.length === 0) {
                        container.innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">No archives found. Add your first archive below.</p>';
                    } else {
                        container.innerHTML = archives.map(formatMobileCard).join('');
                    }
                    
                    container.style.display = 'block';
                    if (tableContainer) tableContainer.style.display = 'none';
                } else {
                    const tbody = document.getElementById('archives-table-body');
                    const mobileContainer = document.getElementById('mobile-archives-container');
                    
                    if (archives.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 40px;">No archives found. Add your first archive below.</td></tr>';
                    } else {
                        tbody.innerHTML = archives.map(formatTableRow).join('');
                    }
                    
                    if (mobileContainer) mobileContainer.style.display = 'none';
                    document.querySelector('.table-container').style.display = 'block';
                }
            } catch (e) {
                console.error('Failed to load archives:', e);
            }
        }
        
        window.addEventListener('resize', function() {
            loadArchives();
        });
        
        async function addArchive(event) {
            event.preventDefault();
            
            const ocrLanguages = [];
            const lang1 = document.getElementById('ocr_lang_1').value;
            const lang2 = document.getElementById('ocr_lang_2').value;
            const lang3 = document.getElementById('ocr_lang_3').value;
            
            if (lang1) ocrLanguages.push(lang1);
            if (lang2) ocrLanguages.push(lang2);
            if (lang3) ocrLanguages.push(lang3);
            
            const ocrLangString = ocrLanguages.join('+') || 'ben';
            
            const formData = {
                book_name: document.getElementById('book_name').value,
                archive_url: document.getElementById('archive_url').value,
                priority: parseInt(document.getElementById('priority').value) || 5,
                pdf_batch_size: parseInt(document.getElementById('pdf_batch_size').value) || 50,
                max_files_per_commit: parseInt(document.getElementById('max_files_per_commit').value) || 50,
                max_pdfs_per_run: parseInt(document.getElementById('max_pdfs_per_run').value) || 20,
                image_zoom: parseFloat(document.getElementById('image_zoom').value) || 3.0,
                image_dpi: parseInt(document.getElementById('image_dpi').value) || 200,
                max_parallel_pdfs: parseInt(document.getElementById('max_parallel_pdfs').value) || 2,
                max_workers: parseInt(document.getElementById('max_workers').value) || 2,
                ocr_oem: parseInt(document.getElementById('ocr_oem').value) || 3,
                ocr_psm: parseInt(document.getElementById('ocr_psm').value) || 3,
                ocr_lang: ocrLangString,
                ocr_workers: parseInt(document.getElementById('ocr_workers').value) || 2,
                metadata: {}
            };
            
            try {
                const response = await fetch('/api/archives', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('Archive added successfully!');
                    document.getElementById('add-archive-form').reset();
                    document.getElementById('priority').value = '5';
                    document.getElementById('pdf_batch_size').value = '50';
                    document.getElementById('max_files_per_commit').value = '50';
                    document.getElementById('max_pdfs_per_run').value = '20';
                    document.getElementById('image_zoom').value = '3.0';
                    document.getElementById('image_dpi').value = '200';
                    document.getElementById('max_parallel_pdfs').value = '2';
                    document.getElementById('max_workers').value = '2';
                    document.getElementById('ocr_oem').value = '3';
                    document.getElementById('ocr_psm').value = '3';
                    document.getElementById('ocr_lang_1').value = 'ben';
                    document.getElementById('ocr_lang_2').value = '';
                    document.getElementById('ocr_lang_3').value = '';
                    document.getElementById('ocr_workers').value = '2';
                    loadArchives();
                } else {
                    alert('Failed to add archive: ' + data.message);
                }
            } catch (e) {
                alert('Failed to add archive');
            }
        }
        
        async function saveConfig(event) {
            event.preventDefault();
            
            const formData = {};
            const form = document.getElementById('config-form');
            
            for (let element of form.elements) {
                if (element.name) {
                    if (element.type === 'number') {
                        formData[element.name] = parseFloat(element.value) || 0;
                    } else {
                        formData[element.name] = element.value;
                    }
                }
            }
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('Configuration saved successfully!');
                    setTimeout(() => location.reload(), 500);
                } else {
                    alert('Failed to save configuration: ' + data.message);
                }
            } catch (e) {
                alert('Failed to save configuration');
            }
        }
        
        async function editArchive(id) {
            try {
                const response = await fetch('/api/archives');
                const archives = await response.json();
                const archive = archives.find(a => a._id === id);
                
                if (archive) {
                    document.getElementById('edit_archive_id').value = id;
                    document.getElementById('edit_book_name').value = archive.book_name || '';
                    document.getElementById('edit_archive_url').value = archive.url || '';
                    document.getElementById('edit_priority').value = archive.priority || 5;
                    document.getElementById('edit_status').value = archive.status || 'pending';
                    
                    const settings = archive.processing_settings || {};
                    document.getElementById('edit_pdf_batch_size').value = settings.pdf_batch_size || 50;
                    document.getElementById('edit_max_files_per_commit').value = settings.max_files_per_commit || 50;
                    document.getElementById('edit_max_pdfs_per_run').value = settings.max_pdfs_per_run || 20;
                    document.getElementById('edit_image_zoom').value = settings.image_zoom || 3.0;
                    document.getElementById('edit_image_dpi').value = settings.image_dpi || 200;
                    document.getElementById('edit_max_parallel_pdfs').value = settings.max_parallel_pdfs || 2;
                    document.getElementById('edit_max_workers').value = settings.max_workers || 2;
                    document.getElementById('edit_ocr_oem').value = settings.ocr_oem || 3;
                    document.getElementById('edit_ocr_psm').value = settings.ocr_psm || 3;
                    document.getElementById('edit_ocr_workers').value = settings.ocr_workers || 2;
                    
                    const ocrLang = settings.ocr_lang || 'ben';
                    const languages = ocrLang.split('+');
                    
                    document.getElementById('edit_ocr_lang_1').value = languages[0] || 'ben';
                    document.getElementById('edit_ocr_lang_2').value = languages[1] || '';
                    document.getElementById('edit_ocr_lang_3').value = languages[2] || '';
                    
                    document.getElementById('edit-modal').style.display = 'flex';
                }
            } catch (e) {
                alert('Failed to load archive details');
            }
        }
        
        function closeEditModal() {
            document.getElementById('edit-modal').style.display = 'none';
        }
        
        async function updateArchive(event) {
            event.preventDefault();
            
            const id = document.getElementById('edit_archive_id').value;
            
            const ocrLanguages = [];
            const lang1 = document.getElementById('edit_ocr_lang_1').value;
            const lang2 = document.getElementById('edit_ocr_lang_2').value;
            const lang3 = document.getElementById('edit_ocr_lang_3').value;
            
            if (lang1) ocrLanguages.push(lang1);
            if (lang2) ocrLanguages.push(lang2);
            if (lang3) ocrLanguages.push(lang3);
            
            const ocrLangString = ocrLanguages.join('+') || 'ben';
            
            const formData = {
                book_name: document.getElementById('edit_book_name').value,
                archive_url: document.getElementById('edit_archive_url').value,
                priority: parseInt(document.getElementById('edit_priority').value),
                status: document.getElementById('edit_status').value,
                pdf_batch_size: parseInt(document.getElementById('edit_pdf_batch_size').value),
                max_files_per_commit: parseInt(document.getElementById('edit_max_files_per_commit').value),
                max_pdfs_per_run: parseInt(document.getElementById('edit_max_pdfs_per_run').value),
                image_zoom: parseFloat(document.getElementById('edit_image_zoom').value),
                image_dpi: parseInt(document.getElementById('edit_image_dpi').value),
                max_parallel_pdfs: parseInt(document.getElementById('edit_max_parallel_pdfs').value),
                max_workers: parseInt(document.getElementById('edit_max_workers').value),
                ocr_oem: parseInt(document.getElementById('edit_ocr_oem').value),
                ocr_psm: parseInt(document.getElementById('edit_ocr_psm').value),
                ocr_lang: ocrLangString,
                ocr_workers: parseInt(document.getElementById('edit_ocr_workers').value)
            };
            
            try {
                const response = await fetch('/api/archives/' + id, {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('Archive updated successfully!');
                    closeEditModal();
                    loadArchives();
                } else {
                    alert('Failed to update archive: ' + data.message);
                }
            } catch (e) {
                alert('Failed to update archive');
            }
        }
        
        async function deleteArchive(id) {
            if (!confirm('Are you sure you want to delete this archive?')) return;
            
            try {
                const response = await fetch('/api/archives/' + id, {
                    method: 'DELETE'
                });
                const data = await response.json();
                if (data.success) {
                    alert('Archive deleted successfully!');
                    loadArchives();
                } else {
                    alert('Failed to delete archive: ' + data.message);
                }
            } catch (e) {
                alert('Failed to delete archive');
            }
        }
        
        function getSelectedIds() {
            if (isMobile()) {
                alert('Bulk actions are only available on desktop view');
                return [];
            }
            return Array.from(document.querySelectorAll('#archives-table-body input[type="checkbox"]:checked'))
                .map(cb => cb.value);
        }
        
        async function processSelected() {
            const selected = getSelectedIds();
            if (selected.length === 0) return;
            
            try {
                for (const id of selected) {
                    await fetch('/api/archives/' + id, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({status: 'pending'})
                    });
                }
                alert('Selected archives marked as pending');
                loadArchives();
            } catch (e) {
                alert('Failed to process selected archives');
            }
        }
        
        async function resetSelected() {
            const selected = getSelectedIds();
            if (selected.length === 0) return;
            
            if (!confirm('Reset selected failed archives?')) return;
            
            try {
                for (const id of selected) {
                    await fetch('/api/archives/' + id, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({status: 'pending', retry_count: 0})
                    });
                }
                alert('Selected archives reset');
                loadArchives();
            } catch (e) {
                alert('Failed to reset archives');
            }
        }
        
        async function deleteSelected() {
            const selected = getSelectedIds();
            if (selected.length === 0) return;
            
            if (!confirm(`Are you sure you want to delete ${selected.length} archives?`)) return;
            
            try {
                for (const id of selected) {
                    await fetch('/api/archives/' + id, {method: 'DELETE'});
                }
                alert('Selected archives deleted');
                loadArchives();
            } catch (e) {
                alert('Failed to delete archives');
            }
        }
        
        async function loadMonitorStatus() {
            try {
                const response = await fetch('/api/monitor/status');
                const data = await response.json();
                
                document.getElementById('mongodb-status').innerHTML = data.mongodb.connected ? 
                    `✅ ${data.mongodb.message}` : `❌ ${data.mongodb.message}`;
                document.getElementById('hf-status').innerHTML = data.hf.connected ? 
                    `✅ ${data.hf.message}` : `❌ ${data.hf.message}`;
                document.getElementById('active-tasks').innerHTML = data.active_tasks || 0;
                document.getElementById('total-completed').innerHTML = data.total_completed || 0;
                document.getElementById('total-pending').innerHTML = data.total_pending || 0;
                document.getElementById('total-failed').innerHTML = data.total_failed || 0;
            } catch (e) {
                console.error('Failed to load monitor status:', e);
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            loadArchives();
            setInterval(loadArchives, 30000);
        });
    </script>
</body>
</html>
"""

# ============ Routes ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard"""

    config = config_manager.get_config()

    html_content = HTML_HEADER + f"""
        <div class="header">
            <h1>📚 তাফসীর PDF প্রসেসর</h1>
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="showTab('config-tab')">⚙️ কনফিগারেশন</button>
                <button class="nav-tab" onclick="showTab('archives-tab')">📁 আর্কাইভ</button>
                <button class="nav-tab" onclick="showTab('monitor-tab')">📊 মনিটর</button>
            </div>
        </div>
        
        <!-- Configuration Tab -->
        <div id="config-tab" class="tab-content active">
            <h2>⚙️ সিস্টেম কনফিগারেশন</h2>
            <p style="color: #666; margin-bottom: 25px; font-size: 16px;">MongoDB এবং Pinecone কনফিগারেশন সেট করুন</p>
            
            <form id="config-form" onsubmit="saveConfig(event)">
                <div class="form-grid">
                    <div class="form-group full-width">
                        <h3>🗄️ MongoDB কনফিগারেশন</h3>
                    </div>
                    
                    <div class="form-group full-width">
                        <label for="mongodb_uri">MongoDB কানেকশন URI *</label>
                        <input type="text" id="mongodb_uri" name="mongodb_uri" value="{config.get('mongodb_uri', '')}" placeholder="mongodb+srv://username:password@cluster.mongodb.net/" required>
                        <button type="button" class="btn btn-secondary" onclick="testMongoDB()" style="margin-top: 12px;">🔌 সংযোগ পরীক্ষা</button>
                        <div id="mongodb-test-result"></div>
                    </div>
                    
                    <div class="form-group">
                        <label for="mongodb_db">ডাটাবেসের নাম</label>
                        <input type="text" id="mongodb_db" name="mongodb_db" value="{config.get('mongodb_db', 'tafsir_db')}" placeholder="tafsir_db">
                    </div>
                    
                    <div class="form-group">
                        <label for="mongodb_collection">কালেকশনের নাম</label>
                        <input type="text" id="mongodb_collection" name="mongodb_collection" value="{config.get('mongodb_collection', 'archive_links')}" placeholder="archive_links">
                    </div>
                    
                    <div class="form-group full-width">
                        <h3>🌲 Pinecone কনফিগারেশন</h3>
                    </div>
                    
                    <div class="form-group full-width">
                        <label for="pinecone_api_key">Pinecone API কী *</label>
                        <input type="password" id="pinecone_api_key" name="pinecone_api_key" value="{config.get('pinecone_api_key', '')}" placeholder="pcsk_..." required>
                    </div>
                    
                    <div class="form-group">
                        <label for="pinecone_index_name">Pinecone ইনডেক্স নাম</label>
                        <input type="text" id="pinecone_index_name" name="pinecone_index_name" value="{config.get('pinecone_index_name', 'tafsir-ocr')}" placeholder="tafsir-ocr">
                    </div>
                    
                    <div class="form-group full-width">
                        <h3>⚡ ডিফল্ট সেটিংস</h3>
                    </div>
                    
                    <div class="form-group">
                        <label for="priority_default">ডিফল্ট অগ্রাধিকার</label>
                        <input type="number" id="priority_default" name="priority_default" value="{config.get('priority_default', 5)}" min="1" max="10">
                        <small>1 = সর্বনিম্ন, 10 = সর্বোচ্চ</small>
                    </div>
                </div>
                
                <div class="btn-group">
                    <button type="submit" class="btn btn-primary">💾 কনফিগারেশন সংরক্ষণ</button>
                    <button type="button" class="btn btn-secondary" onclick="location.reload()">🔄 রিসেট</button>
                </div>
            </form>
        </div>
        
        <!-- Archives Management Tab -->
        <div id="archives-tab" class="tab-content">
            <h2>📁 ইন্টারনেট আর্কাইভ ব্যবস্থাপনা</h2>
            <p style="color: #666; margin-bottom: 25px; font-size: 16px;">নতুন আর্কাইভ যোগ করুন এবং প্রসেসিং সেটিংস কনফিগার করুন</p>
            
            <div class="card">
                <h3>➕ নতুন আর্কাইভ যোগ করুন</h3>
                <form id="add-archive-form" onsubmit="addArchive(event)">
                    
                    <h4>📋 সাধারণ তথ্য</h4>
                    <div style="display: grid; grid-template-columns: 2fr 4fr 1fr; gap: 20px; align-items: end;">
                        <div>
                            <label for="book_name">📚 বইয়ের নাম *</label>
                            <input type="text" id="book_name" placeholder="তাফসীর ফী যিলালিল কোরআন" required style="padding: 14px;">
                        </div>
                        <div>
                            <label for="archive_url">🔗 আর্কাইভ URL *</label>
                            <input type="text" id="archive_url" placeholder="https://archive.org/details/..." required style="padding: 14px;">
                        </div>
                        <div>
                            <label for="priority">⚡ অগ্রাধিকার (1-10)</label>
                            <input type="number" id="priority" value="5" min="1" max="10" style="padding: 14px;">
                        </div>
                    </div>
                    
                    <h4>⚙️ প্রসেসিং সেটিংস</h4>
                    <div class="settings-section">
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                            <div>
                                <label for="pdf_batch_size">📦 PDF ব্যাচ</label>
                                <input type="number" id="pdf_batch_size" value="50" min="10" max="200" style="padding: 12px;">
                                <small>একসাথে কনভার্ট (10-200)</small>
                            </div>
                            <div>
                                <label for="max_files_per_commit">📤 সর্বোচ্চ ফাইল</label>
                                <input type="number" id="max_files_per_commit" value="50" min="10" max="100" style="padding: 12px;">
                                <small>প্রতি কমিটে (10-100)</small>
                            </div>
                            <div>
                                <label for="max_pdfs_per_run">📚 সর্বোচ্চ PDF</label>
                                <input type="number" id="max_pdfs_per_run" value="20" min="1" max="100" style="padding: 12px;">
                                <small>প্রতি রানে (1-100)</small>
                            </div>
                            <div>
                                <label for="image_zoom">🔍 জুম</label>
                                <input type="number" id="image_zoom" value="3.0" min="1.0" max="5.0" step="0.5" style="padding: 12px;">
                                <small>1.0 - 5.0</small>
                            </div>
                            <div>
                                <label for="image_dpi">🖼️ DPI</label>
                                <input type="number" id="image_dpi" value="200" min="72" max="400" style="padding: 12px;">
                                <small>72 - 400 DPI</small>
                            </div>
                            <div>
                                <label for="max_parallel_pdfs">⚡ প্যারালাল</label>
                                <input type="number" id="max_parallel_pdfs" value="2" min="1" max="5" style="padding: 12px;">
                                <small>একসাথে PDF (1-5)</small>
                            </div>
                            <div>
                                <label for="max_workers">🔄 ওয়ার্কার</label>
                                <input type="number" id="max_workers" value="2" min="1" max="5" style="padding: 12px;">
                                <small>ডাউনলোড থ্রেড (1-5)</small>
                            </div>
                        </div>
                    </div>
                    
                    <h4>🔤 OCR সেটিংস</h4>
                    <div class="settings-section">
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                            <div>
                                <label for="ocr_lang_1">ভাষা 1 (প্রাথমিক) *</label>
                                <select id="ocr_lang_1" style="padding: 12px;">
                                    <option value="ben" selected>🇧🇩 বাংলা</option>
                                    <option value="ara">🇸🇦 আরবি</option>
                                    <option value="eng">🇬🇧 ইংরেজি</option>
                                    <option value="urd">🇵🇰 উর্দু</option>
                                    <option value="fas">🇮🇷 ফারসি</option>
                                    <option value="hin">🇮🇳 হিন্দি</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_lang_2">ভাষা 2 (ঐচ্ছিক)</label>
                                <select id="ocr_lang_2" style="padding: 12px;">
                                    <option value="" selected>-- কোনোটি নয় --</option>
                                    <option value="ben">🇧🇩 বাংলা</option>
                                    <option value="ara">🇸🇦 আরবি</option>
                                    <option value="eng">🇬🇧 ইংরেজি</option>
                                    <option value="urd">🇵🇰 উর্দু</option>
                                    <option value="fas">🇮🇷 ফারসি</option>
                                    <option value="hin">🇮🇳 হিন্দি</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_lang_3">ভাষা 3 (ঐচ্ছিক)</label>
                                <select id="ocr_lang_3" style="padding: 12px;">
                                    <option value="" selected>-- কোনোটি নয় --</option>
                                    <option value="ben">🇧🇩 বাংলা</option>
                                    <option value="ara">🇸🇦 আরবি</option>
                                    <option value="eng">🇬🇧 ইংরেজি</option>
                                    <option value="urd">🇵🇰 উর্দু</option>
                                    <option value="fas">🇮🇷 ফারসি</option>
                                    <option value="hin">🇮🇳 হিন্দি</option>
                                </select>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 15px;">
                            <div>
                                <label for="ocr_oem">OCR ইঞ্জিন মোড</label>
                                <select id="ocr_oem" style="padding: 12px;">
                                    <option value="3" selected>3 - LSTM নিউরাল (সেরা)</option>
                                    <option value="1">1 - শুধু LSTM</option>
                                    <option value="2">2 - LSTM + লিগ্যাসি</option>
                                    <option value="0">0 - শুধু লিগ্যাসি</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_psm">পৃষ্ঠা সেগমেন্টেশন</label>
                                <select id="ocr_psm" style="padding: 12px;">
                                    <option value="3" selected>3 - স্বয়ংক্রিয়</option>
                                    <option value="6">6 - সমান টেক্সট ব্লক</option>
                                    <option value="1">1 - OSD সহ স্বয়ংক্রিয়</option>
                                    <option value="4">4 - একক কলাম</option>
                                    <option value="7">7 - একক টেক্সট লাইন</option>
                                    <option value="8">8 - একক শব্দ</option>
                                    <option value="11">11 - বিক্ষিপ্ত টেক্সট</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_workers">OCR ওয়ার্কার</label>
                                <input type="number" id="ocr_workers" value="2" min="1" max="4" style="padding: 12px;">
                                <small>প্যারালাল থ্রেড (1-4)</small>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px; display: flex; justify-content: flex-end;">
                        <button type="submit" class="btn btn-primary" style="padding: 14px 32px;">➕ আর্কাইভ যোগ করুন</button>
                    </div>
                </form>
            </div>
            
            <!-- Mobile Card View Container -->
            <div id="mobile-archives-container" style="display: none;"></div>
            
            <!-- Desktop Table View -->
            <div class="table-container">
                <h3>📚 বিদ্যমান আর্কাইভসমূহ</h3>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 40px;"><input type="checkbox" id="select-all-checkbox" onclick="document.querySelectorAll('#archives-table-body input[type=checkbox]').forEach(cb=>cb.checked=this.checked)"></th>
                            <th>বইয়ের নাম</th>
                            <th>URL</th>
                            <th>অবস্থা</th>
                            <th>অগ্রাধিকার</th>
                            <th>ব্যাচ</th>
                            <th>অগ্রগতি</th>
                            <th>সর্বশেষ আপডেট</th>
                            <th style="width: 100px;">কার্যক্রম</th>
                        </tr>
                    </thead>
                    <tbody id="archives-table-body">
                        <tr><td colspan="9" style="text-align: center; padding: 40px;">আর্কাইভ লোড হচ্ছে...</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="btn-group">
                <button class="btn btn-success" onclick="processSelected()">▶️ নির্বাচিত প্রক্রিয়া করুন</button>
                <button class="btn btn-warning" onclick="resetSelected()">🔄 ব্যর্থ রিসেট</button>
                <button class="btn btn-danger" onclick="deleteSelected()">🗑️ নির্বাচিত মুছুন</button>
            </div>
        </div>
        
        <!-- Monitor Tab -->
        <div id="monitor-tab" class="tab-content">
            <h2>📊 সিস্টেম মনিটর</h2>
            <p style="color: #666; margin-bottom: 25px; font-size: 16px;">সিস্টেম অবস্থা এবং সংযোগ পর্যবেক্ষণ</p>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 25px;">
                <div style="background: linear-gradient(135deg, #1a5f7a, #0d3b4c); color: white; padding: 30px; border-radius: 15px;">
                    <h3 style="color: white;">🗄️ MongoDB অবস্থা</h3>
                    <div id="mongodb-status" style="font-size: 16px; margin-top: 15px;">পরীক্ষা করতে মনিটর ট্যাবে ক্লিক করুন</div>
                </div>
                
                <div style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; padding: 30px; border-radius: 15px;">
                    <h3 style="color: white;">🌲 Pinecone অবস্থা</h3>
                    <div id="hf-status" style="font-size: 16px; margin-top: 15px;">পরীক্ষা করতে মনিটর ট্যাবে ক্লিক করুন</div>
                </div>
            </div>
            
            <div style="margin-top: 30px; background: white; padding: 25px; border-radius: 15px; border: 1px solid #e0e0e0;">
                <h3>📈 প্রসেসিং পরিসংখ্যান</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 20px;" class="grid">
                    <div style="text-align: center;">
                        <div style="font-size: 36px; font-weight: bold; color: #1a5f7a;" id="active-tasks">0</div>
                        <div style="color: #666;">সক্রিয় কাজ</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 36px; font-weight: bold; color: #28a745;" id="total-completed">0</div>
                        <div style="color: #666;">সম্পন্ন</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 36px; font-weight: bold; color: #ffc107;" id="total-pending">0</div>
                        <div style="color: #666;">অপেক্ষমান</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 36px; font-weight: bold; color: #dc3545;" id="total-failed">0</div>
                        <div style="color: #666;">ব্যর্থ</div>
                    </div>
                </div>
            </div>
        </div>
    """ + HTML_FOOTER

    return HTMLResponse(content=html_content)

# ============ API Routes ============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/config")
async def get_config():
    return config_manager.get_config()

@app.post("/api/config")
async def save_config(config: SystemConfig):
    """Save configuration"""
    print(f"[API] Saving configuration...")

    config_dict = config.dict()

    if config.mongodb_uri:
        print(f"[API] Initializing MongoDB...")
        init_success = config_manager.initialize(config.mongodb_uri, "tafsir_config")
        if init_success:
            print(f"[API] MongoDB initialized successfully")
        else:
            print(f"[API] MongoDB initialization failed")

    success = config_manager.save_config(config_dict)

    if success:
        return {"success": True, "message": "Configuration saved successfully"}
    else:
        return {"success": False, "message": "Failed to save configuration"}

@app.post("/api/test/mongodb")
async def test_mongodb(request: Request):
    data = await request.json()
    uri = data.get("uri", "")
    if not uri:
        return {"success": False, "message": "URI is required"}
    success, message = config_manager.test_mongodb_connection(uri)
    return {"success": success, "message": message}

@app.get("/api/archives")
async def get_archives():
    """Get all archive items"""
    return config_manager.get_archives()

@app.post("/api/archives")
async def add_archive(item: ArchiveItem):
    """Add a new archive item"""
    archive_data = {
        "book_name": item.book_name,
        "url": item.archive_url,
        "status": "pending",
        "priority": item.priority,
        "retry_count": 0,
        "metadata": item.metadata or {},
        "processing_settings": {
            "pdf_batch_size": item.pdf_batch_size,
            "max_files_per_commit": item.max_files_per_commit,
            "max_pdfs_per_run": item.max_pdfs_per_run,
            "image_zoom": item.image_zoom,
            "image_dpi": item.image_dpi,
            "max_parallel_pdfs": item.max_parallel_pdfs,
            "max_workers": item.max_workers,
            "ocr_oem": item.ocr_oem,
            "ocr_psm": item.ocr_psm,
            "ocr_lang": item.ocr_lang,
            "ocr_workers": item.ocr_workers
        }
    }

    success, result = config_manager.add_archive(archive_data)

    if success:
        return {"success": True, "message": "Archive added successfully", "id": result}
    else:
        return {"success": False, "message": result}

@app.put("/api/archives/{archive_id}")
async def update_archive(archive_id: str, item: ArchiveUpdateModel):
    """Update an archive item"""
    update_data = {}

    if item.book_name is not None:
        update_data["book_name"] = item.book_name
    if item.archive_url is not None:
        update_data["url"] = item.archive_url
    if item.priority is not None:
        update_data["priority"] = item.priority
    if item.status is not None:
        update_data["status"] = item.status
        if item.status == "pending":
            update_data["retry_count"] = 0
    if item.retry_count is not None:
        update_data["retry_count"] = item.retry_count
    if item.pdf_batch_size is not None:
        update_data["processing_settings.pdf_batch_size"] = item.pdf_batch_size
    if item.max_files_per_commit is not None:
        update_data["processing_settings.max_files_per_commit"] = item.max_files_per_commit
    if item.max_pdfs_per_run is not None:
        update_data["processing_settings.max_pdfs_per_run"] = item.max_pdfs_per_run
    if item.image_zoom is not None:
        update_data["processing_settings.image_zoom"] = item.image_zoom
    if item.image_dpi is not None:
        update_data["processing_settings.image_dpi"] = item.image_dpi
    if item.max_parallel_pdfs is not None:
        update_data["processing_settings.max_parallel_pdfs"] = item.max_parallel_pdfs
    if item.max_workers is not None:
        update_data["processing_settings.max_workers"] = item.max_workers
    if item.ocr_oem is not None:
        update_data["processing_settings.ocr_oem"] = item.ocr_oem
    if item.ocr_psm is not None:
        update_data["processing_settings.ocr_psm"] = item.ocr_psm
    if item.ocr_lang is not None:
        update_data["processing_settings.ocr_lang"] = item.ocr_lang
    if item.ocr_workers is not None:
        update_data["processing_settings.ocr_workers"] = item.ocr_workers

    success, message = config_manager.update_archive(archive_id, update_data)
    return {"success": success, "message": message}

@app.delete("/api/archives/{archive_id}")
async def delete_archive(archive_id: str):
    """Delete an archive item"""
    success, message = config_manager.delete_archive(archive_id)
    return {"success": success, "message": message}

@app.get("/api/monitor/status")
async def get_system_status():
    """Get system status for monitoring"""
    config = config_manager.get_config()

    mongodb_status = {"connected": False, "message": "Not configured"}
    if config_manager.is_connected:
        mongodb_status = {"connected": True, "message": "Connected to MongoDB"}
    elif config.get("mongodb_uri"):
        success, message = config_manager.test_mongodb_connection(config["mongodb_uri"])
        if success:
            config_manager.initialize(config["mongodb_uri"], "tafsir_config")
            mongodb_status = {"connected": True, "message": message}
        else:
            mongodb_status = {"connected": False, "message": message}

    hf_status = {"connected": False, "message": "Pinecone not configured"}

    stats = config_manager.get_statistics()

    return {
        "mongodb": mongodb_status,
        "hf": hf_status,
        **stats
    }

# ============ Startup Event ============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("[Startup] Initializing...")

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            if config.get("mongodb_uri"):
                print(f"[Startup] Found saved config, connecting to MongoDB...")
                config_manager.initialize(config["mongodb_uri"], "tafsir_config")
        except Exception as e:
            print(f"[Startup] Failed to load config: {e}")

    print("[Startup] Initialization complete")

# ============ Main ============

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)