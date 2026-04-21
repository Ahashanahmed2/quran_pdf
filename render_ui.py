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
    "priority_default": ৫
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
    priority_default: int = Field(৫, ge=১, le=১০, description="Default priority for new archives")

class ArchiveItem(BaseModel):
    """Archive item to process"""
    book_name: str = Field(..., description="Book name in Bengali/English")
    archive_url: str = Field(..., description="Internet Archive URL")
    priority: int = Field(৫, ge=১, le=১০, description="Processing priority (১-১০)")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

    # Processing settings for this specific archive
    pdf_batch_size: int = Field(৫০, ge=১০, le=২০০)
    max_files_per_commit: int = Field(৫০, ge=১০, le=১০০)
    max_pdfs_per_run: int = Field(২০, ge=১, le=১০০)
    image_zoom: float = Field(৩.০, ge=১.০, le=৫.০)
    image_dpi: int = Field(২০০, ge=৭২, le=৪০০)
    max_parallel_pdfs: int = Field(২, ge=১, le=৫)
    max_workers: int = Field(২, ge=১, le=৫)
    
    # OCR Settings
    ocr_oem: int = Field(৩, ge=০, le=৩)
    ocr_psm: int = Field(৩, ge=০, le=১৩)
    ocr_lang: str = Field("ben", description="OCR Languages (e.g., ben+ara+eng)")
    ocr_workers: int = Field(২, ge=১, le=৪)

class ArchiveUpdateModel(BaseModel):
    """Update archive item"""
    book_name: Optional[str] = None
    archive_url: Optional[str] = None
    priority: Optional[int] = Field(None, ge=১, le=১০)
    status: Optional[str] = None
    metadata: Optional[Dict] = None
    pdf_batch_size: Optional[int] = Field(None, ge=১০, le=২০০)
    max_files_per_commit: Optional[int] = Field(None, ge=১০, le=১০০)
    max_pdfs_per_run: Optional[int] = Field(None, ge=১, le=১০০)
    image_zoom: Optional[float] = Field(None, ge=১.০, le=৫.০)
    image_dpi: Optional[int] = Field(None, ge=৭২, le=৪০০)
    max_parallel_pdfs: Optional[int] = Field(None, ge=১, le=৫)
    max_workers: Optional[int] = Field(None, ge=১, le=৫)
    ocr_oem: Optional[int] = Field(None, ge=০, le=৩)
    ocr_psm: Optional[int] = Field(None, ge=০, le=১৩)
    ocr_lang: Optional[str] = None
    ocr_workers: Optional[int] = Field(None, ge=১, le=৪)
    retry_count: Optional[int] = Field(None, ge=০, le=১০)

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
            self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=১০০০০)
            self.client.admin.command('ping')

            self.db = self.client[db_name]
            self.config_collection = self.db["system_config"]

            # Archive collection-ও initialize করুন
            data_db_name = DEFAULT_CONFIG.get("mongodb_db", "tafsir_db")
            data_collection_name = DEFAULT_CONFIG.get("mongodb_collection", "archive_links")
            data_db = self.client[data_db_name]
            self.archive_collection = data_db[data_collection_name]

            self.is_connected = True
            print(f"[ConfigManager] ✅ Connected to MongoDB: {db_name}")
            return True

        except Exception as e:
            print(f"[ConfigManager] ❌ MongoDB connection failed: {e}")
            self.is_connected = False
            return False

    def get_config(self) -> Dict:
        """Get current configuration"""
        if self.is_connected and self.config_collection is not None:
            try:
                config = self.config_collection.find_one({"_id": "current"})
                if config:
                    config.pop("_id", None)
                    print("[ConfigManager] ✅ Loaded config from MongoDB")
                    return config
            except Exception as e:
                print(f"[ConfigManager] Failed to load from MongoDB: {e}")

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    print("[ConfigManager] ✅ Loaded config from file")
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
                print(f"[ConfigManager] ✅ Saved to MongoDB. Modified: {result.modified_count}, Upserted: {result.upserted_id}")

                saved = self.config_collection.find_one({"_id": "current"})
                if saved:
                    print(f"[ConfigManager] ✅ Verified - Config exists in MongoDB")

                    # MongoDB URI সেভ হলে archive collection রি-ইনিশিয়ালাইজ
                    if config.get("mongodb_uri") and config.get("mongodb_db") and config.get("mongodb_collection"):
                        try:
                            data_db = self.client[config["mongodb_db"]]
                            self.archive_collection = data_db[config["mongodb_collection"]]
                            print(f"[ConfigManager] ✅ Archive collection initialized: {config['mongodb_db']}.{config['mongodb_collection']}")
                        except Exception as e:
                            print(f"[ConfigManager] ⚠️ Failed to initialize archive collection: {e}")

                    return True

            except Exception as e:
                print(f"[ConfigManager] ❌ Failed to save to MongoDB: {e}")
        else:
            print("[ConfigManager] ⚠️ Not connected to MongoDB, saving to file only")

        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=২, default=str)
            print(f"[ConfigManager] ✅ Saved to file: {CONFIG_FILE}")
            return True
        except Exception as e:
            print(f"[ConfigManager] ❌ Failed to save to file: {e}")
            return False

    def test_mongodb_connection(self, uri: str) -> tuple:
        """Test MongoDB connection"""
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=৫০০০)
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

    def get_archives(self, limit: int = ১০০) -> List[Dict]:
        """Get all archive items"""
        if not self.is_connected or self.archive_collection is None:
            print("[ConfigManager] ⚠️ Not connected to archive collection")
            return []

        try:
            archives = list(self.archive_collection.find().sort("created_at", -১).limit(limit))
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
            if result.deleted_count > ০:
                return True, "Archive deleted successfully"
            else:
                return False, "Archive not found"
        except Exception as e:
            return False, str(e)

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        if not self.is_connected or self.archive_collection is None:
            return {
                "active_tasks": ০,
                "total_completed": ০,
                "total_pending": ০,
                "total_failed": ০
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
                "active_tasks": ০,
                "total_completed": ০,
                "total_pending": ০,
                "total_failed": ০
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

app = FastAPI(title="তাফসীর PDF প্রসেসর কনফিগ", version="৩.০.০")

# Initialize config manager
config_manager = ConfigManager()

# ============ HTML Templates ============

HTML_HEADER = """
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=১.০, maximum-scale=১.০, user-scalable=yes">
    <title>তাফসীর PDF প্রসেসর - কনফিগারেশন</title>
    <style>
        * { margin: ০; padding: ০; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(১৩৫deg, #১a৫f৭a ০%, #০d৩b৪c ১০০%);
            min-height: ১০০vh;
            padding: ২০px;
        }
        .container {
            max-width: ১৬০০px;
            margin: ০ auto;
        }
        .header {
            background: white;
            border-radius: ১৫px;
            padding: ২০px ৩০px;
            margin-bottom: ২০px;
            box-shadow: ০ ৫px ১৫px rgba(০,০,০,০.১);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            color: #১a৫f৭a;
            font-size: ২৪px;
        }
        .nav-tabs {
            display: flex;
            gap: ১০px;
        }
        .nav-tab {
            padding: ১২px ২৪px;
            background: #f০f০f০;
            border: none;
            border-radius: ৮px;
            cursor: pointer;
            font-size: ১৫px;
            font-weight: ৫০০;
            transition: all ০.৩s;
        }
        .nav-tab:hover {
            background: #e০e০e০;
        }
        .nav-tab.active {
            background: #১a৫f৭a;
            color: white;
        }
        .tab-content {
            background: white;
            border-radius: ১৫px;
            padding: ৩০px;
            box-shadow: ০ ৫px ১৫px rgba(০,০,০,০.১);
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(২, ১fr);
            gap: ২৫px;
        }
        .form-group {
            margin-bottom: ২০px;
        }
        .form-group.full-width {
            grid-column: span ২;
        }
        label {
            display: block;
            margin-bottom: ৮px;
            font-weight: ৬০০;
            color: #৩৩৩;
            font-size: ১৬px;
        }
        input, select, textarea {
            width: ১০০%;
            padding: ১৪px ১৮px;
            border: ২px solid #e০e০e০;
            border-radius: ১০px;
            font-size: ১৬px;
            transition: border-color ০.৩s, box-shadow ০.৩s;
            background: white;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #১a৫f৭a;
            box-shadow: ০ ০ ০ ৪px rgba(২৬, ৯৫, ১২২, ০.১);
        }
        input[type="number"] {
            -moz-appearance: textfield;
        }
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: ০;
        }
        .btn {
            padding: ১৪px ২৮px;
            border: none;
            border-radius: ১০px;
            font-size: ১৬px;
            font-weight: ৬০০;
            cursor: pointer;
            transition: all ০.৩s;
            white-space: nowrap;
        }
        .btn-primary {
            background: linear-gradient(১৩৫deg, #১a৫f৭a ০%, #০d৩b৪c ১০০%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-২px);
            box-shadow: ০ ৫px ১৫px rgba(২৬, ৯৫, ১২২, ০.৪);
        }
        .btn-secondary {
            background: #৬c৭৫৭d;
            color: white;
        }
        .btn-secondary:hover {
            background: #৫a৬২৬৮;
        }
        .btn-success {
            background: #২৮a৭৪৫;
            color: white;
        }
        .btn-danger {
            background: #dc৩৫৪৫;
            color: white;
        }
        .btn-warning {
            background: #ffc১০৭;
            color: #৩৩৩;
        }
        .btn-group {
            display: flex;
            gap: ১৫px;
            margin-top: ২৫px;
            flex-wrap: wrap;
        }
        .status-badge {
            display: inline-block;
            padding: ৬px ১৪px;
            border-radius: ২০px;
            font-size: ১৩px;
            font-weight: bold;
        }
        .status-pending { background: #fff৩cd; color: #৮৫৬৪০৪; }
        .status-processing { background: #cce৫ff; color: #০০৪০৮৫; }
        .status-completed { background: #d৪edda; color: #১৫৫৭২৪; }
        .status-failed { background: #f৮d৭da; color: #৭২১c২৪; }
        
        .table-container {
            overflow-x: auto;
            margin-top: ৩০px;
            border-radius: ১২px;
            border: ১px solid #e০e০e০;
        }
        table {
            width: ১০০%;
            border-collapse: collapse;
            font-size: ১৪px;
        }
        th, td {
            padding: ১৪px ১৬px;
            text-align: left;
            border-bottom: ১px solid #e০e০e০;
        }
        th {
            background: #f৮f৯fa;
            font-weight: ৬০০;
            color: #৩৩৩;
            font-size: ১৪px;
        }
        td {
            color: #৫৫৫;
        }
        tr:hover {
            background: #f৮f৯fa;
        }
        .alert {
            padding: ১৮px ২০px;
            border-radius: ১০px;
            margin-bottom: ২০px;
            font-size: ১৫px;
        }
        .alert-success {
            background: #d৪edda;
            color: #১৫৫৭২৪;
            border: ১px solid #c৩e৬cb;
        }
        .alert-error {
            background: #f৮d৭da;
            color: #৭২১c২৪;
            border: ১px solid #f৫c৬cb;
        }
        h2 {
            font-size: ২৪px;
            margin-bottom: ১৫px;
            color: #৩৩৩;
        }
        h3 {
            font-size: ১৮px;
            margin-bottom: ১৫px;
            color: #৪৪৪;
        }
        h4 {
            font-size: ১৬px;
            margin: ২০px ০ ১৫px ০;
            color: #৫৫৫;
            border-bottom: ২px solid #e০e০e০;
            padding-bottom: ১০px;
        }
        .card {
            background: #f৮f৯fa;
            padding: ২৫px;
            border-radius: ১৫px;
            margin-bottom: ৩০px;
        }
        small {
            display: block;
            margin-top: ৫px;
            color: #৬৬৬;
            font-size: ১৩px;
        }
        .settings-section {
            background: white;
            padding: ২০px;
            border-radius: ১০px;
            margin-top: ২০px;
            border: ১px solid #e০e০e০;
        }
        .edit-form-container {
            position: fixed;
            top: ০;
            left: ০;
            right: ০;
            bottom: ০;
            background: rgba(০,০,০,০.৫);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: ১০০০;
        }
        .edit-form {
            background: white;
            padding: ৩০px;
            border-radius: ১৫px;
            max-width: ৯০০px;
            max-height: ৯০vh;
            overflow-y: auto;
            width: ৯০%;
        }
        
        .mobile-archive-card {
            background: #f৮f৯fa;
            padding: ১৫px;
            border-radius: ১০px;
            margin-bottom: ১৫px;
            border-left: ৪px solid #১a৫f৭a;
        }
        
        @media screen and (max-width: ৭৬৮px) {
            body {
                padding: ১০px;
            }
            
            .header {
                flex-direction: column;
                gap: ১৫px;
                padding: ১৫px ২০px;
            }
            
            .header h1 {
                font-size: ২০px;
                text-align: center;
            }
            
            .nav-tabs {
                flex-wrap: wrap;
                justify-content: center;
                width: ১০০%;
            }
            
            .nav-tab {
                padding: ১০px ১৬px;
                font-size: ১৪px;
                flex: ১ ০ auto;
            }
            
            .tab-content {
                padding: ২০px ১৫px;
            }
            
            .form-grid {
                grid-template-columns: ১fr !important;
                gap: ১৫px;
            }
            
            .form-group.full-width {
                grid-column: span ১;
            }
            
            #add-archive-form > div:first-of-type {
                grid-template-columns: ১fr !important;
                gap: ১৫px;
            }
            
            .settings-section > div {
                grid-template-columns: ১fr !important;
                gap: ১৫px;
            }
            
            .btn {
                padding: ১২px ১৬px;
                font-size: ১৪px;
                width: ১০০%;
            }
            
            .btn-group {
                flex-direction: column;
                gap: ১০px;
            }
            
            .card {
                padding: ১৫px;
            }
            
            .edit-form {
                padding: ২০px;
                width: ৯৫%;
            }
            
            #monitor-tab > div:first-of-type {
                grid-template-columns: ১fr !important;
                gap: ১৫px;
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
                <div style="display: grid; gap: ১৫px;">
                    <div>
                        <label for="edit_book_name">📚 বইয়ের নাম *</label>
                        <input type="text" id="edit_book_name" required style="padding: ১৪px;">
                    </div>
                    <div>
                        <label for="edit_archive_url">🔗 আর্কাইভ URL *</label>
                        <input type="text" id="edit_archive_url" required style="padding: ১৪px;">
                    </div>
                    <div style="display: grid; grid-template-columns: ১fr ১fr; gap: ১৫px;">
                        <div>
                            <label for="edit_priority">⚡ অগ্রাধিকার (১-১০)</label>
                            <input type="number" id="edit_priority" min="১" max="১০" style="padding: ১৪px;">
                        </div>
                        <div>
                            <label for="edit_status">📊 অবস্থা</label>
                            <select id="edit_status" style="padding: ১৪px;">
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
                <div style="display: grid; grid-template-columns: repeat(২, ১fr); gap: ১৫px;">
                    <div>
                        <label for="edit_pdf_batch_size">📦 PDF ব্যাচ সাইজ</label>
                        <input type="number" id="edit_pdf_batch_size" min="১০" max="২০০" style="padding: ১২px;">
                    </div>
                    <div>
                        <label for="edit_max_files_per_commit">📤 প্রতি কমিটে সর্বোচ্চ ফাইল</label>
                        <input type="number" id="edit_max_files_per_commit" min="১০" max="১০০" style="padding: ১২px;">
                    </div>
                    <div>
                        <label for="edit_max_pdfs_per_run">📚 প্রতি রানে সর্বোচ্চ PDF</label>
                        <input type="number" id="edit_max_pdfs_per_run" min="১" max="১০০" style="padding: ১২px;">
                    </div>
                    <div>
                        <label for="edit_image_zoom">🔍 ইমেজ জুম</label>
                        <input type="number" id="edit_image_zoom" min="১.০" max="৫.০" step="০.৫" style="padding: ১২px;">
                    </div>
                    <div>
                        <label for="edit_image_dpi">🖼️ ইমেজ DPI</label>
                        <input type="number" id="edit_image_dpi" min="৭২" max="৪০০" style="padding: ১২px;">
                    </div>
                    <div>
                        <label for="edit_max_parallel_pdfs">⚡ প্যারালাল PDF</label>
                        <input type="number" id="edit_max_parallel_pdfs" min="১" max="৫" style="padding: ১২px;">
                    </div>
                    <div>
                        <label for="edit_max_workers">🔄 ডাউনলোড ওয়ার্কার</label>
                        <input type="number" id="edit_max_workers" min="১" max="৫" style="padding: ১২px;">
                    </div>
                </div>
                
                <h4>🔤 OCR সেটিংস</h4>
                <div style="display: grid; grid-template-columns: repeat(৩, ১fr); gap: ১৫px;">
                    <div>
                        <label for="edit_ocr_lang_১">ভাষা ১ (প্রাথমিক)</label>
                        <select id="edit_ocr_lang_১" style="padding: ১২px;">
                            <option value="ben" selected>🇧🇩 বাংলা</option>
                            <option value="ara">🇸🇦 আরবি</option>
                            <option value="eng">🇬🇧 ইংরেজি</option>
                            <option value="urd">🇵🇰 উর্দু</option>
                            <option value="fas">🇮🇷 ফারসি</option>
                            <option value="hin">🇮🇳 হিন্দি</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_lang_২">ভাষা ২ (ঐচ্ছিক)</label>
                        <select id="edit_ocr_lang_২" style="padding: ১২px;">
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
                        <label for="edit_ocr_lang_৩">ভাষা ৩ (ঐচ্ছিক)</label>
                        <select id="edit_ocr_lang_৩" style="padding: ১২px;">
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
                <div style="display: grid; grid-template-columns: repeat(৩, ১fr); gap: ১৫px; margin-top: ১৫px;">
                    <div>
                        <label for="edit_ocr_oem">OCR ইঞ্জিন মোড</label>
                        <select id="edit_ocr_oem" style="padding: ১২px;">
                            <option value="৩" selected>৩ - LSTM নিউরাল (সেরা)</option>
                            <option value="১">১ - শুধু LSTM</option>
                            <option value="২">২ - LSTM + লিগ্যাসি</option>
                            <option value="০">০ - শুধু লিগ্যাসি</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_psm">পৃষ্ঠা সেগমেন্টেশন</label>
                        <select id="edit_ocr_psm" style="padding: ১২px;">
                            <option value="৩" selected>৩ - স্বয়ংক্রিয়</option>
                            <option value="৬">৬ - সমান টেক্সট ব্লক</option>
                            <option value="১">১ - OSD সহ স্বয়ংক্রিয়</option>
                            <option value="৪">৪ - একক কলাম</option>
                            <option value="৭">৭ - একক টেক্সট লাইন</option>
                            <option value="৮">৮ - একক শব্দ</option>
                            <option value="১১">১১ - বিক্ষিপ্ত টেক্সট</option>
                            <option value="১২">১২ - OSD সহ বিক্ষিপ্ত</option>
                            <option value="১৩">১৩ - র' লাইন</option>
                        </select>
                    </div>
                    <div>
                        <label for="edit_ocr_workers">OCR ওয়ার্কার</label>
                        <input type="number" id="edit_ocr_workers" min="১" max="৪" value="২" style="padding: ১২px;">
                        <small>প্যারালাল OCR থ্রেড</small>
                    </div>
                </div>
                
                <div class="btn-group" style="margin-top: ২৫px;">
                    <button type="submit" class="btn btn-primary">💾 আপডেট করুন</button>
                    <button type="button" class="btn btn-secondary" onclick="closeEditModal()">❌ বাতিল</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        function isMobile() {
            return window.innerWidth <= ৭৬৮;
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
            resultDiv.innerHTML = 'পরীক্ষা করা হচ্ছে...';
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
                resultDiv.innerHTML = `<div class="alert alert-error">❌ সংযোগ ব্যর্থ</div>`;
            }
        }
        
        async function testHF() {
            const token = document.getElementById('hf_token').value;
            const resultDiv = document.getElementById('hf-test-result');
            resultDiv.innerHTML = 'পরীক্ষা করা হচ্ছে...';
            try {
                const response = await fetch('/api/test/hf', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({token: token})
                });
                const data = await response.json();
                if (data.success) {
                    resultDiv.innerHTML = `<div class="alert alert-success">✅ ${data.message}</div>`;
                } else {
                    resultDiv.innerHTML = `<div class="alert alert-error">❌ ${data.message}</div>`;
                }
            } catch (e) {
                resultDiv.innerHTML = `<div class="alert alert-error">❌ সংযোগ ব্যর্থ</div>`;
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
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: ১০px;">
                        <strong style="font-size: ১৬px;">${archive.book_name || 'N/A'}</strong>
                        <span class="status-badge status-${archive.status || 'pending'}">${archive.status || 'অপেক্ষমান'}</span>
                    </div>
                    <div style="font-size: ১২px; color: #৬৬৬; margin-bottom: ৮px; word-break: break-all;">
                        ${archive.url ? archive.url.substring(০, ৫০) + '...' : 'N/A'}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(৩, ১fr); gap: ৮px; font-size: ১২px; margin-bottom: ১০px;">
                        <div><span style="color: #৬৬৬;">অগ্রাধিকার:</span> ${archive.priority || ৫}</div>
                        <div><span style="color: #৬৬৬;">ব্যাচ:</span> ${archive.processing_settings?.pdf_batch_size || ৫০}</div>
                        <div><span style="color: #৬৬৬;">অগ্রগতি:</span> ${archive.completed_pdfs || ০}/${archive.total_pdfs || ০}</div>
                    </div>
                    <div style="display: flex; gap: ৮px; justify-content: flex-end;">
                        <button class="btn btn-secondary" onclick="editArchive('${archive._id}')" style="padding: ৮px ১৬px; font-size: ১২px;">✏️ সম্পাদনা</button>
                        <button class="btn btn-danger" onclick="deleteArchive('${archive._id}')" style="padding: ৮px ১৬px; font-size: ১২px;">🗑️ মুছুন</button>
                    </div>
                </div>
            `;
        }
        
        function formatTableRow(archive) {
            return `
                <tr>
                    <td><input type="checkbox" value="${archive._id}" style="width: ১৮px; height: ১৮px;"></td>
                    <td><strong>${archive.book_name || 'N/A'}</strong></td>
                    <td>${archive.url ? archive.url.substring(০, ৪০) + '...' : 'N/A'}</td>
                    <td><span class="status-badge status-${archive.status || 'pending'}">${archive.status || 'অপেক্ষমান'}</span></td>
                    <td>${archive.priority || ৫}</td>
                    <td>${archive.processing_settings?.pdf_batch_size || ৫০}</td>
                    <td>${archive.completed_pdfs || ০}/${archive.total_pdfs || ০}</td>
                    <td>${archive.updated_at ? new Date(archive.updated_at).toLocaleString('bn-BD') : 'N/A'}</td>
                    <td>
                        <button class="btn btn-secondary" onclick="editArchive('${archive._id}')" style="padding: ৬px ১২px; font-size: ১২px;">✏️</button>
                        <button class="btn btn-danger" onclick="deleteArchive('${archive._id}')" style="padding: ৬px ১২px; font-size: ১২px;">🗑️</button>
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
                    
                    if (archives.length === ০) {
                        container.innerHTML = '<p style="text-align: center; padding: ৪০px; color: #৬৬৬;">কোনো আর্কাইভ পাওয়া যায়নি। নিচে প্রথম আর্কাইভ যোগ করুন।</p>';
                    } else {
                        container.innerHTML = archives.map(formatMobileCard).join('');
                    }
                    
                    container.style.display = 'block';
                    if (tableContainer) tableContainer.style.display = 'none';
                } else {
                    const tbody = document.getElementById('archives-table-body');
                    const mobileContainer = document.getElementById('mobile-archives-container');
                    
                    if (archives.length === ০) {
                        tbody.innerHTML = '<tr><td colspan="৯" style="text-align: center; padding: ৪০px;">কোনো আর্কাইভ পাওয়া যায়নি। নিচে প্রথম আর্কাইভ যোগ করুন।</td></tr>';
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
            
            // Collect OCR languages
            const ocrLanguages = [];
            const lang১ = document.getElementById('ocr_lang_১').value;
            const lang২ = document.getElementById('ocr_lang_২').value;
            const lang৩ = document.getElementById('ocr_lang_৩').value;
            
            if (lang১) ocrLanguages.push(lang১);
            if (lang২) ocrLanguages.push(lang২);
            if (lang৩) ocrLanguages.push(lang৩);
            
            const ocrLangString = ocrLanguages.join('+') || 'ben';
            
            const formData = {
                book_name: document.getElementById('book_name').value,
                archive_url: document.getElementById('archive_url').value,
                priority: parseInt(document.getElementById('priority').value) || ৫,
                pdf_batch_size: parseInt(document.getElementById('pdf_batch_size').value) || ৫০,
                max_files_per_commit: parseInt(document.getElementById('max_files_per_commit').value) || ৫০,
                max_pdfs_per_run: parseInt(document.getElementById('max_pdfs_per_run').value) || ২০,
                image_zoom: parseFloat(document.getElementById('image_zoom').value) || ৩.০,
                image_dpi: parseInt(document.getElementById('image_dpi').value) || ২০০,
                max_parallel_pdfs: parseInt(document.getElementById('max_parallel_pdfs').value) || ২,
                max_workers: parseInt(document.getElementById('max_workers').value) || ২,
                ocr_oem: parseInt(document.getElementById('ocr_oem').value) || ৩,
                ocr_psm: parseInt(document.getElementById('ocr_psm').value) || ৩,
                ocr_lang: ocrLangString,
                ocr_workers: parseInt(document.getElementById('ocr_workers').value) || ২,
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
                    alert('✅ ' + data.message);
                    document.getElementById('add-archive-form').reset();
                    document.getElementById('priority').value = '৫';
                    document.getElementById('pdf_batch_size').value = '৫০';
                    document.getElementById('max_files_per_commit').value = '৫০';
                    document.getElementById('max_pdfs_per_run').value = '২০';
                    document.getElementById('image_zoom').value = '৩.০';
                    document.getElementById('image_dpi').value = '২০০';
                    document.getElementById('max_parallel_pdfs').value = '২';
                    document.getElementById('max_workers').value = '২';
                    document.getElementById('ocr_oem').value = '৩';
                    document.getElementById('ocr_psm').value = '৩';
                    document.getElementById('ocr_lang_১').value = 'ben';
                    document.getElementById('ocr_lang_২').value = '';
                    document.getElementById('ocr_lang_৩').value = '';
                    document.getElementById('ocr_workers').value = '২';
                    loadArchives();
                } else {
                    alert('❌ ' + data.message);
                }
            } catch (e) {
                alert('❌ আর্কাইভ যোগ করতে ব্যর্থ');
            }
        }
        
        async function saveConfig(event) {
            event.preventDefault();
            
            const formData = {};
            const form = document.getElementById('config-form');
            
            for (let element of form.elements) {
                if (element.name) {
                    if (element.type === 'number') {
                        formData[element.name] = parseFloat(element.value) || ০;
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
                    alert('✅ কনফিগারেশন সফলভাবে সংরক্ষিত!');
                    setTimeout(() => location.reload(), ৫০০);
                } else {
                    alert('❌ কনফিগারেশন সংরক্ষণ ব্যর্থ: ' + data.message);
                }
            } catch (e) {
                alert('❌ কনফিগারেশন সংরক্ষণ ব্যর্থ');
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
                    document.getElementById('edit_priority').value = archive.priority || ৫;
                    document.getElementById('edit_status').value = archive.status || 'pending';
                    
                    const settings = archive.processing_settings || {};
                    document.getElementById('edit_pdf_batch_size').value = settings.pdf_batch_size || ৫০;
                    document.getElementById('edit_max_files_per_commit').value = settings.max_files_per_commit || ৫০;
                    document.getElementById('edit_max_pdfs_per_run').value = settings.max_pdfs_per_run || ২০;
                    document.getElementById('edit_image_zoom').value = settings.image_zoom || ৩.০;
                    document.getElementById('edit_image_dpi').value = settings.image_dpi || ২০০;
                    document.getElementById('edit_max_parallel_pdfs').value = settings.max_parallel_pdfs || ২;
                    document.getElementById('edit_max_workers').value = settings.max_workers || ২;
                    document.getElementById('edit_ocr_oem').value = settings.ocr_oem || ৩;
                    document.getElementById('edit_ocr_psm').value = settings.ocr_psm || ৩;
                    document.getElementById('edit_ocr_workers').value = settings.ocr_workers || ২;
                    
                    // Parse OCR languages
                    const ocrLang = settings.ocr_lang || 'ben';
                    const languages = ocrLang.split('+');
                    
                    document.getElementById('edit_ocr_lang_১').value = languages[০] || 'ben';
                    document.getElementById('edit_ocr_lang_২').value = languages[১] || '';
                    document.getElementById('edit_ocr_lang_৩').value = languages[২] || '';
                    
                    document.getElementById('edit-modal').style.display = 'flex';
                }
            } catch (e) {
                alert('আর্কাইভ বিবরণ লোড করতে ব্যর্থ');
            }
        }
        
        function closeEditModal() {
            document.getElementById('edit-modal').style.display = 'none';
        }
        
        async function updateArchive(event) {
            event.preventDefault();
            
            const id = document.getElementById('edit_archive_id').value;
            
            // Collect OCR languages
            const ocrLanguages = [];
            const lang১ = document.getElementById('edit_ocr_lang_১').value;
            const lang২ = document.getElementById('edit_ocr_lang_২').value;
            const lang৩ = document.getElementById('edit_ocr_lang_৩').value;
            
            if (lang১) ocrLanguages.push(lang১);
            if (lang২) ocrLanguages.push(lang২);
            if (lang৩) ocrLanguages.push(lang৩);
            
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
                    alert('✅ ' + data.message);
                    closeEditModal();
                    loadArchives();
                } else {
                    alert('❌ ' + data.message);
                }
            } catch (e) {
                alert('❌ আর্কাইভ আপডেট করতে ব্যর্থ');
            }
        }
        
        async function deleteArchive(id) {
            if (!confirm('আপনি কি নিশ্চিত এই আর্কাইভটি মুছে ফেলতে চান?')) return;
            
            try {
                const response = await fetch('/api/archives/' + id, {
                    method: 'DELETE'
                });
                const data = await response.json();
                if (data.success) {
                    alert('✅ আর্কাইভ সফলভাবে মুছে ফেলা হয়েছে!');
                    loadArchives();
                } else {
                    alert('❌ ' + data.message);
                }
            } catch (e) {
                alert('❌ আর্কাইভ মুছতে ব্যর্থ');
            }
        }
        
        function getSelectedIds() {
            if (isMobile()) {
                alert('বাল্ক অপারেশন শুধুমাত্র ডেস্কটপ ভিউতে উপলব্ধ');
                return [];
            }
            return Array.from(document.querySelectorAll('#archives-table-body input[type="checkbox"]:checked'))
                .map(cb => cb.value);
        }
        
        async function processSelected() {
            const selected = getSelectedIds();
            if (selected.length === ০) return;
            
            try {
                for (const id of selected) {
                    await fetch('/api/archives/' + id, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({status: 'pending'})
                    });
                }
                alert('✅ নির্বাচিত আর্কাইভগুলো অপেক্ষমান হিসেবে চিহ্নিত করা হয়েছে');
                loadArchives();
            } catch (e) {
                alert('❌ নির্বাচিত আর্কাইভ প্রক্রিয়া করতে ব্যর্থ');
            }
        }
        
        async function resetSelected() {
            const selected = getSelectedIds();
            if (selected.length === ০) return;
            
            if (!confirm('ব্যর্থ আর্কাইভগুলো রিসেট করতে চান?')) return;
            
            try {
                for (const id of selected) {
                    await fetch('/api/archives/' + id, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({status: 'pending', retry_count: ০})
                    });
                }
                alert('✅ নির্বাচিত আর্কাইভ রিসেট করা হয়েছে');
                loadArchives();
            } catch (e) {
                alert('❌ আর্কাইভ রিসেট করতে ব্যর্থ');
            }
        }
        
        async function deleteSelected() {
            const selected = getSelectedIds();
            if (selected.length === ০) return;
            
            if (!confirm(`আপনি কি নিশ্চিত ${selected.length}টি আর্কাইভ মুছে ফেলতে চান?`)) return;
            
            try {
                for (const id of selected) {
                    await fetch('/api/archives/' + id, {method: 'DELETE'});
                }
                alert('✅ নির্বাচিত আর্কাইভ মুছে ফেলা হয়েছে');
                loadArchives();
            } catch (e) {
                alert('❌ আর্কাইভ মুছতে ব্যর্থ');
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
                document.getElementById('active-tasks').innerHTML = data.active_tasks || ০;
                document.getElementById('total-completed').innerHTML = data.total_completed || ০;
                document.getElementById('total-pending').innerHTML = data.total_pending || ০;
                document.getElementById('total-failed').innerHTML = data.total_failed || ০;
            } catch (e) {
                console.error('Failed to load monitor status:', e);
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            loadArchives();
            setInterval(loadArchives, ৩০০০০);
        });
    </script>
</body>
</html>
"""

# ============ Routes ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """প্রধান ড্যাশবোর্ড"""

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
            <p style="color: #৬৬৬; margin-bottom: ২৫px; font-size: ১৬px;">MongoDB এবং Pinecone কনফিগারেশন সেট করুন</p>
            
            <form id="config-form" onsubmit="saveConfig(event)">
                <div class="form-grid">
                    <div class="form-group full-width">
                        <h3>🗄️ MongoDB কনফিগারেশন</h3>
                    </div>
                    
                    <div class="form-group full-width">
                        <label for="mongodb_uri">MongoDB কানেকশন URI *</label>
                        <input type="text" id="mongodb_uri" name="mongodb_uri" value="{config.get('mongodb_uri', '')}" placeholder="mongodb+srv://username:password@cluster.mongodb.net/" required>
                        <button type="button" class="btn btn-secondary" onclick="testMongoDB()" style="margin-top: ১২px;">🔌 সংযোগ পরীক্ষা</button>
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
                        <input type="number" id="priority_default" name="priority_default" value="{config.get('priority_default', ৫)}" min="১" max="১০">
                        <small>১ = সর্বনিম্ন, ১০ = সর্বোচ্চ</small>
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
            <p style="color: #৬৬৬; margin-bottom: ২৫px; font-size: ১৬px;">নতুন আর্কাইভ যোগ করুন এবং প্রসেসিং সেটিংস কনফিগার করুন</p>
            
            <div class="card">
                <h3>➕ নতুন আর্কাইভ যোগ করুন</h3>
                <form id="add-archive-form" onsubmit="addArchive(event)">
                    
                    <h4>📋 সাধারণ তথ্য</h4>
                    <div style="display: grid; grid-template-columns: ২fr ৪fr ১fr; gap: ২০px; align-items: end;">
                        <div>
                            <label for="book_name">📚 বইয়ের নাম *</label>
                            <input type="text" id="book_name" placeholder="তাফসীর ফী যিলালিল কোরআন" required style="padding: ১৪px;">
                        </div>
                        <div>
                            <label for="archive_url">🔗 আর্কাইভ URL *</label>
                            <input type="text" id="archive_url" placeholder="https://archive.org/details/..." required style="padding: ১৪px;">
                        </div>
                        <div>
                            <label for="priority">⚡ অগ্রাধিকার (১-১০)</label>
                            <input type="number" id="priority" value="৫" min="১" max="১০" style="padding: ১৪px;">
                        </div>
                    </div>
                    
                    <h4>⚙️ প্রসেসিং সেটিংস</h4>
                    <div class="settings-section">
                        <div style="display: grid; grid-template-columns: repeat(৪, ১fr); gap: ২০px;">
                            <div>
                                <label for="pdf_batch_size">📦 PDF ব্যাচ</label>
                                <input type="number" id="pdf_batch_size" value="৫০" min="১০" max="২০০" style="padding: ১২px;">
                                <small>একসাথে কনভার্ট (১০-২০০)</small>
                            </div>
                            <div>
                                <label for="max_files_per_commit">📤 সর্বোচ্চ ফাইল</label>
                                <input type="number" id="max_files_per_commit" value="৫০" min="১০" max="১০০" style="padding: ১২px;">
                                <small>প্রতি কমিটে (১০-১০০)</small>
                            </div>
                            <div>
                                <label for="max_pdfs_per_run">📚 সর্বোচ্চ PDF</label>
                                <input type="number" id="max_pdfs_per_run" value="২০" min="১" max="১০০" style="padding: ১২px;">
                                <small>প্রতি রানে (১-১০০)</small>
                            </div>
                            <div>
                                <label for="image_zoom">🔍 জুম</label>
                                <input type="number" id="image_zoom" value="৩.০" min="১.০" max="৫.০" step="০.৫" style="padding: ১২px;">
                                <small>১.০ - ৫.০</small>
                            </div>
                            <div>
                                <label for="image_dpi">🖼️ DPI</label>
                                <input type="number" id="image_dpi" value="২০০" min="৭২" max="৪০০" style="padding: ১২px;">
                                <small>৭২ - ৪০০ DPI</small>
                            </div>
                            <div>
                                <label for="max_parallel_pdfs">⚡ প্যারালাল</label>
                                <input type="number" id="max_parallel_pdfs" value="২" min="১" max="৫" style="padding: ১২px;">
                                <small>একসাথে PDF (১-৫)</small>
                            </div>
                            <div>
                                <label for="max_workers">🔄 ওয়ার্কার</label>
                                <input type="number" id="max_workers" value="২" min="১" max="৫" style="padding: ১২px;">
                                <small>ডাউনলোড থ্রেড (১-৫)</small>
                            </div>
                        </div>
                    </div>
                    
                    <h4>🔤 OCR সেটিংস</h4>
                    <div class="settings-section">
                        <div style="display: grid; grid-template-columns: repeat(৩, ১fr); gap: ২০px;">
                            <div>
                                <label for="ocr_lang_১">ভাষা ১ (প্রাথমিক) *</label>
                                <select id="ocr_lang_১" style="padding: ১২px;">
                                    <option value="ben" selected>🇧🇩 বাংলা</option>
                                    <option value="ara">🇸🇦 আরবি</option>
                                    <option value="eng">🇬🇧 ইংরেজি</option>
                                    <option value="urd">🇵🇰 উর্দু</option>
                                    <option value="fas">🇮🇷 ফারসি</option>
                                    <option value="hin">🇮🇳 হিন্দি</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_lang_২">ভাষা ২ (ঐচ্ছিক)</label>
                                <select id="ocr_lang_২" style="padding: ১২px;">
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
                                <label for="ocr_lang_৩">ভাষা ৩ (ঐচ্ছিক)</label>
                                <select id="ocr_lang_৩" style="padding: ১২px;">
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
                        <div style="display: grid; grid-template-columns: repeat(৩, ১fr); gap: ২০px; margin-top: ১৫px;">
                            <div>
                                <label for="ocr_oem">OCR ইঞ্জিন মোড</label>
                                <select id="ocr_oem" style="padding: ১২px;">
                                    <option value="৩" selected>৩ - LSTM নিউরাল (সেরা)</option>
                                    <option value="১">১ - শুধু LSTM</option>
                                    <option value="২">২ - LSTM + লিগ্যাসি</option>
                                    <option value="০">০ - শুধু লিগ্যাসি</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_psm">পৃষ্ঠা সেগমেন্টেশন</label>
                                <select id="ocr_psm" style="padding: ১২px;">
                                    <option value="৩" selected>৩ - স্বয়ংক্রিয়</option>
                                    <option value="৬">৬ - সমান টেক্সট ব্লক</option>
                                    <option value="১">১ - OSD সহ স্বয়ংক্রিয়</option>
                                    <option value="৪">৪ - একক কলাম</option>
                                    <option value="৭">৭ - একক টেক্সট লাইন</option>
                                    <option value="৮">৮ - একক শব্দ</option>
                                    <option value="১১">১১ - বিক্ষিপ্ত টেক্সট</option>
                                </select>
                            </div>
                            <div>
                                <label for="ocr_workers">OCR ওয়ার্কার</label>
                                <input type="number" id="ocr_workers" value="২" min="১" max="৪" style="padding: ১২px;">
                                <small>প্যারালাল থ্রেড (১-৪)</small>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: ২০px; display: flex; justify-content: flex-end;">
                        <button type="submit" class="btn btn-primary" style="padding: ১৪px ৩২px;">➕ আর্কাইভ যোগ করুন</button>
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
                            <th style="width: ৪০px;"><input type="checkbox" id="select-all-checkbox" onclick="document.querySelectorAll('#archives-table-body input[type=checkbox]').forEach(cb=>cb.checked=this.checked)"></th>
                            <th>বইয়ের নাম</th>
                            <th>URL</th>
                            <th>অবস্থা</th>
                            <th>অগ্রাধিকার</th>
                            <th>ব্যাচ</th>
                            <th>অগ্রগতি</th>
                            <th>সর্বশেষ আপডেট</th>
                            <th style="width: ১০০px;">কার্যক্রম</th>
                        </tr>
                    </thead>
                    <tbody id="archives-table-body">
                        <tr><td colspan="৯" style="text-align: center; padding: ৪০px;">আর্কাইভ লোড হচ্ছে...</td></tr>
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
            <p style="color: #৬৬৬; margin-bottom: ২৫px; font-size: ১৬px;">সিস্টেম অবস্থা এবং সংযোগ পর্যবেক্ষণ</p>
            
            <div style="display: grid; grid-template-columns: repeat(২, ১fr); gap: ২৫px;">
                <div style="background: linear-gradient(১৩৫deg, #১a৫f৭a, #০d৩b৪c); color: white; padding: ৩০px; border-radius: ১৫px;">
                    <h3 style="color: white;">🗄️ MongoDB অবস্থা</h3>
                    <div id="mongodb-status" style="font-size: ১৬px; margin-top: ১৫px;">পরীক্ষা করতে মনিটর ট্যাবে ক্লিক করুন</div>
                </div>
                
                <div style="background: linear-gradient(১৩৫deg, #f০৯৩fb, #f৫৫৭৬c); color: white; padding: ৩০px; border-radius: ১৫px;">
                    <h3 style="color: white;">🌲 Pinecone অবস্থা</h3>
                    <div id="hf-status" style="font-size: ১৬px; margin-top: ১৫px;">পরীক্ষা করতে মনিটর ট্যাবে ক্লিক করুন</div>
                </div>
            </div>
            
            <div style="margin-top: ৩০px; background: white; padding: ২৫px; border-radius: ১৫px; border: ১px solid #e০e০e০;">
                <h3>📈 প্রসেসিং পরিসংখ্যান</h3>
                <div style="display: grid; grid-template-columns: repeat(৪, ১fr); gap: ২০px; margin-top: ২০px;" class="grid">
                    <div style="text-align: center;">
                        <div style="font-size: ৩৬px; font-weight: bold; color: #১a৫f৭a;" id="active-tasks">০</div>
                        <div style="color: #৬৬৬;">সক্রিয় কাজ</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: ৩৬px; font-weight: bold; color: #২৮a৭৪৫;" id="total-completed">০</div>
                        <div style="color: #৬৬৬;">সম্পন্ন</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: ৩৬px; font-weight: bold; color: #ffc১০৭;" id="total-pending">০</div>
                        <div style="color: #৬৬৬;">অপেক্ষমান</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: ৩৬px; font-weight: bold; color: #dc৩৫৪৫;" id="total-failed">০</div>
                        <div style="color: #৬৬৬;">ব্যর্থ</div>
                    </div>
                </div>
            </div>
        </div>
    """ + HTML_FOOTER

    return HTMLResponse(content=html_content)

# ============ API Routes ============

@app.get("/health")
async def health_check():
    """স্বাস্থ্য পরীক্ষা এন্ডপয়েন্ট"""
    return {"status": "সুস্থ", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/config")
async def get_config():
    return config_manager.get_config()

@app.post("/api/config")
async def save_config(config: SystemConfig):
    """কনফিগারেশন সংরক্ষণ"""
    print(f"[API] কনফিগারেশন সংরক্ষণ হচ্ছে...")

    config_dict = config.dict()

    if config.mongodb_uri:
        print(f"[API] MongoDB ইনিশিয়ালাইজ হচ্ছে...")
        init_success = config_manager.initialize(config.mongodb_uri, "tafsir_config")
        if init_success:
            print(f"[API] ✅ MongoDB সফলভাবে ইনিশিয়ালাইজ হয়েছে")
        else:
            print(f"[API] ❌ MongoDB ইনিশিয়ালাইজেশন ব্যর্থ")

    success = config_manager.save_config(config_dict)

    if success:
        return {"success": True, "message": "কনফিগারেশন সফলভাবে সংরক্ষিত"}
    else:
        return {"success": False, "message": "কনফিগারেশন সংরক্ষণ ব্যর্থ"}

@app.post("/api/test/mongodb")
async def test_mongodb(request: Request):
    data = await request.json()
    uri = data.get("uri", "")
    if not uri:
        return {"success": False, "message": "URI প্রয়োজন"}
    success, message = config_manager.test_mongodb_connection(uri)
    return {"success": success, "message": message}

@app.get("/api/archives")
async def get_archives():
    """সকল আর্কাইভ আইটেম পান"""
    return config_manager.get_archives()

@app.post("/api/archives")
async def add_archive(item: ArchiveItem):
    """নতুন আর্কাইভ আইটেম যোগ করুন"""
    archive_data = {
        "book_name": item.book_name,
        "url": item.archive_url,
        "status": "pending",
        "priority": item.priority,
        "retry_count": ০,
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
        return {"success": True, "message": "আর্কাইভ সফলভাবে যোগ করা হয়েছে", "id": result}
    else:
        return {"success": False, "message": result}

@app.put("/api/archives/{archive_id}")
async def update_archive(archive_id: str, item: ArchiveUpdateModel):
    """আর্কাইভ আইটেম আপডেট করুন"""
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
            update_data["retry_count"] = ০
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
    """আর্কাইভ আইটেম মুছুন"""
    success, message = config_manager.delete_archive(archive_id)
    return {"success": success, "message": message}

@app.get("/api/monitor/status")
async def get_system_status():
    """মনিটরিংয়ের জন্য সিস্টেম অবস্থা"""
    config = config_manager.get_config()

    mongodb_status = {"connected": False, "message": "কনফিগার করা হয়নি"}
    if config_manager.is_connected:
        mongodb_status = {"connected": True, "message": "MongoDB-তে সংযুক্ত"}
    elif config.get("mongodb_uri"):
        success, message = config_manager.test_mongodb_connection(config["mongodb_uri"])
        if success:
            config_manager.initialize(config["mongodb_uri"], "tafsir_config")
            mongodb_status = {"connected": True, "message": message}
        else:
            mongodb_status = {"connected": False, "message": message}

    hf_status = {"connected": False, "message": "Pinecone কনফিগার করা হয়নি"}

    stats = config_manager.get_statistics()

    return {
        "mongodb": mongodb_status,
        "hf": hf_status,
        **stats
    }

# ============ Startup Event ============

@app.on_event("startup")
async def startup_event():
    """স্টার্টআপে ইনিশিয়ালাইজ"""
    print("[Startup] ইনিশিয়ালাইজ হচ্ছে...")

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            if config.get("mongodb_uri"):
                print(f"[Startup] সংরক্ষিত কনফিগারেশন পাওয়া গেছে, MongoDB-তে সংযোগ হচ্ছে...")
                config_manager.initialize(config["mongodb_uri"], "tafsir_config")
        except Exception as e:
            print(f"[Startup] কনফিগারেশন লোড করতে ব্যর্থ: {e}")

    print("[Startup] ✅ ইনিশিয়ালাইজেশন সম্পূর্ণ")

# ============ Main ============

if __name__ == "__main__":
    port = int(os.environ.get("PORT", ৮০০০))
    uvicorn.run(app, host="০.০.০.০", port=port)