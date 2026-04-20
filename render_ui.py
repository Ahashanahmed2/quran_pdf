#!/usr/bin/env python3
"""
Render.com Web UI for Tafsir PDF Processor Configuration
Complete management interface for MongoDB, HF, and GitHub Actions
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
from pymongo.errors import PyMongoError
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
    "github_repo": "",
    "github_token": "",
    "pdf_batch_size": 100,
    "max_files_per_commit": 100,
    "max_pdfs_per_run": 50,
    "image_zoom": 4.0,
    "image_dpi": 300,
    "cron_schedule": "0 */6 * * *",
    "priority_default": 5
}

# Local config file for Render.com (fallback)
CONFIG_FILE = Path("/data/config.json") if os.path.exists("/data") else Path("config.json")

# ============ Models ============

class SystemConfig(BaseModel):
    """Complete system configuration"""
    hf_token: str = Field(..., description="HuggingFace API Token")
    hf_dataset: str = Field(..., description="HuggingFace Dataset (username/dataset_name)")
    mongodb_uri: str = Field(..., description="MongoDB Connection URI")
    mongodb_db: str = Field("tafsir_db", description="MongoDB Database Name")
    mongodb_collection: str = Field("archive_links", description="MongoDB Collection Name")
    github_repo: str = Field("", description="GitHub Repository (username/repo)")
    github_token: str = Field("", description="GitHub Personal Access Token")
    pdf_batch_size: int = Field(100, ge=10, le=500, description="PDF batch size")
    max_files_per_commit: int = Field(100, ge=10, le=200, description="Max files per HF commit")
    max_pdfs_per_run: int = Field(50, ge=1, le=200, description="Max PDFs per GitHub Actions run")
    image_zoom: float = Field(4.0, ge=1.0, le=5.0, description="Image zoom factor")
    image_dpi: int = Field(300, ge=72, le=600, description="Image DPI")
    cron_schedule: str = Field("0 */6 * * *", description="Cron schedule for GitHub Actions")
    priority_default: int = Field(5, ge=1, le=10, description="Default priority for new archives")

class ArchiveItem(BaseModel):
    """Archive item to process"""
    book_name: str = Field(..., description="Book name in Bengali/English")
    archive_url: str = Field(..., description="Internet Archive URL")
    priority: int = Field(5, ge=1, le=10, description="Processing priority (1-10)")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

class ArchiveUpdateModel(BaseModel):
    """Update archive item"""
    book_name: Optional[str] = None
    archive_url: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    status: Optional[str] = None
    metadata: Optional[Dict] = None

class BulkArchiveInput(BaseModel):
    """Bulk archive input"""
    items: List[ArchiveItem]

# ============ Config Manager ============

class ConfigManager:
    """Manage system configuration in MongoDB"""
    
    def __init__(self):
        self.config_collection = None
        self.client = None
        
    def initialize(self, mongodb_uri: str = None, db_name: str = "tafsir_config"):
        """Initialize MongoDB connection"""
        if mongodb_uri:
            try:
                self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
                self.client.admin.command('ping')  # Test connection
                self.db = self.client[db_name]
                self.config_collection = self.db["system_config"]
                return True
            except Exception as e:
                print(f"MongoDB connection failed: {e}")
                return False
        return False
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        if self.config_collection:
            config = self.config_collection.find_one({"_id": "current"})
            if config:
                config.pop("_id", None)
                return config
        
        # Fallback to file or defaults
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        
        return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict) -> bool:
        """Save configuration"""
        config["updated_at"] = datetime.utcnow()
        
        if self.config_collection:
            try:
                self.config_collection.update_one(
                    {"_id": "current"},
                    {"$set": config},
                    upsert=True
                )
                return True
            except Exception as e:
                print(f"Failed to save to MongoDB: {e}")
        
        # Fallback to file
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Failed to save to file: {e}")
            return False
    
    def test_mongodb_connection(self, uri: str) -> tuple:
        """Test MongoDB connection"""
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            
            # Get server info
            server_info = client.server_info()
            return True, f"Connected to MongoDB {server_info.get('version', 'Unknown')}"
        except Exception as e:
            return False, str(e)
    
    def test_hf_connection(self, token: str) -> tuple:
        """Test HuggingFace connection"""
        try:
            api = HfApi(token=token)
            # Try to get user info
            user = api.whoami()
            return True, f"Connected as {user.get('name', 'Unknown')}"
        except Exception as e:
            return False, str(e)

# ============ GitHub Actions Generator ============

class GitHubActionsGenerator:
    """Generate GitHub Actions workflow files"""
    
    @staticmethod
    def generate_workflow(config: Dict) -> str:
        """Generate GitHub Actions workflow YAML"""
        
        yaml_content = f"""name: Process Internet Archive PDFs

on:
  schedule:
    - cron: '{config.get("cron_schedule", "0 */6 * * *")}'
  
  workflow_dispatch:
    inputs:
      max_pdfs:
        description: 'Maximum PDFs to process'
        required: false
        default: '{config.get("max_pdfs_per_run", 50)}'
      batch_size:
        description: 'PDF batch size'
        required: false
        default: '{config.get("pdf_batch_size", 100)}'

jobs:
  process:
    runs-on: ubuntu-latest
    timeout-minutes: 360
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Cache pip packages
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('requirements.txt') }}}}
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install PyMuPDF huggingface-hub pymongo dnspython
    
    - name: Download processor script
      run: |
        curl -o tafsir_processor.py https://raw.githubusercontent.com/{config.get("github_repo", "username/repo")}/main/tafsir_processor.py
    
    - name: Run processor
      env:
        HF_TOKEN: ${{{{ secrets.HF_TOKEN }}}}}
        HF_DATASET: ${{{{ secrets.HF_DATASET }}}}}
        MONGODB_URI: ${{{{ secrets.MONGODB_URI }}}}}
        MONGODB_DB: {config.get("mongodb_db", "tafsir_db")}
        MONGODB_COLLECTION: {config.get("mongodb_collection", "archive_links")}
        MAX_PDFS_PER_RUN: ${{{{ github.event.inputs.max_pdfs || '{config.get("max_pdfs_per_run", 50)}' }}}}
        PDF_BATCH_SIZE: ${{{{ github.event.inputs.batch_size || '{config.get("pdf_batch_size", 100)}' }}}}
        MAX_FILES_PER_COMMIT: '{config.get("max_files_per_commit", 100)}'
        IMAGE_ZOOM: '{config.get("image_zoom", 4.0)}'
        IMAGE_DPI: '{config.get("image_dpi", 300)}'
        PDF_SLEEP_BETWEEN: '1'
      run: python tafsir_processor.py
"""
        return yaml_content
    
    @staticmethod
    def generate_secrets_instructions(config: Dict) -> str:
        """Generate secrets setup instructions"""
        
        return f"""## GitHub Secrets Setup

Add these secrets to your GitHub repository:
`{config.get("github_repo", "username/repo")}`

1. Go to: Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add the following secrets:




```

HF_TOKEN={config.get("hf_token", "your_hf_token_here")}
HF_DATASET={config.get("hf_dataset", "username/dataset_name")}
MONGODB_URI={config.get("mongodb_uri", "your_mongodb_uri")}
MONGODB_DB={config.get("mongodb_db", "tafsir_db")}
MONGODB_COLLECTION={config.get("mongodb_collection", "archive_links")}

```









## Verify Setup

After adding secrets, go to Actions tab and run workflow manually to test.
"""

# ============ FastAPI App ============

app = FastAPI(title="Tafsir PDF Processor Config", version="2.0.0")

# Initialize config manager
config_manager = ConfigManager()

# ============ HTML Templates ============

HTML_HEADER = """
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>তাফসীর PDF প্রসেসর - কনফিগারেশন</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
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
            color: #333;
            font-size: 24px;
        }
        .nav-tabs {
            display: flex;
            gap: 10px;
        }
        .nav-tab {
            padding: 10px 20px;
            background: #f0f0f0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        .nav-tab:hover {
            background: #e0e0e0;
        }
        .nav-tab.active {
            background: #667eea;
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
            gap: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .status-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-pending { background: #fff3cd; color: #856404; }
        .status-processing { background: #cce5ff; color: #004085; }
        .status-completed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        
        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
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
        .connection-status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-connected { background: #28a745; }
        .status-disconnected { background: #dc3545; }
        
        .code-block {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
"""

HTML_FOOTER = """
    </div>
    <script>
        // Tab switching
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.nav-tab').forEach(btn => {
                btn.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Test MongoDB connection
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
        
        // Test HF connection
        async function testHF() {
            const token = document.getElementById('hf_token').value;
            const resultDiv = document.getElementById('hf-test-result');
            
            resultDiv.innerHTML = 'Testing...';
            
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
                resultDiv.innerHTML = `<div class="alert alert-error">❌ Connection failed</div>`;
            }
        }
        
        // Generate GitHub Actions workflow
        async function generateWorkflow() {
            const resultDiv = document.getElementById('workflow-result');
            
            try {
                const response = await fetch('/api/generate/workflow');
                const data = await response.json();
                
                resultDiv.innerHTML = `
                    <h3>Generated Workflow File</h3>
                    <div class="code-block">${escapeHtml(data.workflow)}</div>
                    <h3 style="margin-top: 20px;">Secrets Setup Instructions</h3>
                    <div class="code-block">${escapeHtml(data.instructions)}</div>
                `;
            } catch (e) {
                resultDiv.innerHTML = '<div class="alert alert-error">Failed to generate workflow</div>';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load archives
        async function loadArchives() {
            try {
                const response = await fetch('/api/archives');
                const archives = await response.json();
                
                const tbody = document.getElementById('archives-table-body');
                
                if (archives.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No archives found</td></tr>';
                    return;
                }
                
                tbody.innerHTML = archives.map(item => `
                    <tr>
                        <td><input type="checkbox" value="${item._id}"></td>
                        <td>${item.book_name || 'N/A'}</td>
                        <td>${item.url ? item.url.substring(0, 50) + '...' : 'N/A'}</td>
                        <td><span class="status-badge status-${item.status || 'pending'}">${item.status || 'pending'}</span></td>
                        <td>${item.priority || 5}</td>
                        <td>${item.completed_pdfs || 0}/${item.total_pdfs || 0}</td>
                        <td>
                            <button class="btn btn-secondary" onclick="editArchive('${item._id}')" style="padding: 5px 10px;">Edit</button>
                            <button class="btn btn-danger" onclick="deleteArchive('${item._id}')" style="padding: 5px 10px;">Delete</button>
                        </td>
                    </tr>
                `).join('');
            } catch (e) {
                console.error('Failed to load archives:', e);
            }
        }
        
        // Add archive
        async function addArchive(event) {
            event.preventDefault();
            
            const formData = {
                book_name: document.getElementById('book_name').value,
                archive_url: document.getElementById('archive_url').value,
                priority: parseInt(document.getElementById('priority').value),
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
                    loadArchives();
                } else {
                    alert('Failed to add archive: ' + data.message);
                }
            } catch (e) {
                alert('Failed to add archive');
            }
        }
        
        // Save configuration
        async function saveConfig(event) {
            event.preventDefault();
            
            const formData = {};
            const form = document.getElementById('config-form');
            
            for (let element of form.elements) {
                if (element.name) {
                    if (element.type === 'number') {
                        formData[element.name] = parseFloat(element.value);
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
                } else {
                    alert('Failed to save configuration: ' + data.message);
                }
            } catch (e) {
                alert('Failed to save configuration');
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadArchives();
            setInterval(loadArchives, 30000); // Refresh every 30 seconds
        });
    </script>
</body>
</html>
"""

# ============ Routes ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard"""
    
    # Get current config
    config = config_manager.get_config()
    
    # Generate HTML content
    html_content = HTML_HEADER + f"""
        <div class="header">
            <h1>📚 তাফসীর PDF প্রসেসর কনফিগারেশন</h1>
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="showTab('config-tab')">⚙️ কনফিগারেশন</button>
                <button class="nav-tab" onclick="showTab('archives-tab')">📁 আর্কাইভ ম্যানেজমেন্ট</button>
                <button class="nav-tab" onclick="showTab('github-tab')">🔧 GitHub Actions</button>
                <button class="nav-tab" onclick="showTab('monitor-tab')">📊 মনিটরিং</button>
            </div>
        </div>
        
        <!-- Configuration Tab -->
        <div id="config-tab" class="tab-content active">
            <h2>সিস্টেম কনফিগারেশন</h2>
            <p style="color: #666; margin-bottom: 20px;">MongoDB, HuggingFace এবং প্রসেসিং সেটিংস কনফিগার করুন</p>
            
            <form id="config-form" onsubmit="saveConfig(event)">
                <div class="form-grid">
                    <div class="form-group full-width">
                        <h3>🔐 HuggingFace Configuration</h3>
                    </div>
                    
                    <div class="form-group full-width">
                        <label for="hf_token">HuggingFace API Token *</label>
                        <input type="password" id="hf_token" name="hf_token" value="{config.get('hf_token', '')}" required>
                        <button type="button" class="btn btn-secondary" onclick="testHF()" style="margin-top: 10px;">Test Connection</button>
                        <div id="hf-test-result"></div>
                    </div>
                    
                    <div class="form-group full-width">
                        <label for="hf_dataset">HuggingFace Dataset *</label>
                        <input type="text" id="hf_dataset" name="hf_dataset" value="{config.get('hf_dataset', '')}" placeholder="username/dataset_name" required>
                    </div>
                    
                    <div class="form-group full-width">
                        <h3>🗄️ MongoDB Configuration</h3>
                    </div>
                    
                    <div class="form-group full-width">
                        <label for="mongodb_uri">MongoDB Connection URI *</label>
                        <input type="text" id="mongodb_uri" name="mongodb_uri" value="{config.get('mongodb_uri', '')}" placeholder="mongodb+srv://..." required>
                        <button type="button" class="btn btn-secondary" onclick="testMongoDB()" style="margin-top: 10px;">Test Connection</button>
                        <div id="mongodb-test-result"></div>
                    </div>
                    
                    <div class="form-group">
                        <label for="mongodb_db">Database Name</label>
                        <input type="text" id="mongodb_db" name="mongodb_db" value="{config.get('mongodb_db', 'tafsir_db')}">
                    </div>
                    
                    <div class="form-group">
                        <label for="mongodb_collection">Collection Name</label>
                        <input type="text" id="mongodb_collection" name="mongodb_collection" value="{config.get('mongodb_collection', 'archive_links')}">
                    </div>
                    
                    <div class="form-group full-width">
                        <h3>🐙 GitHub Configuration</h3>
                    </div>
                    
                    <div class="form-group">
                        <label for="github_repo">GitHub Repository</label>
                        <input type="text" id="github_repo" name="github_repo" value="{config.get('github_repo', '')}" placeholder="username/repository">
                    </div>
                    
                    <div class="form-group">
                        <label for="github_token">GitHub Token (Optional)</label>
                        <input type="password" id="github_token" name="github_token" value="{config.get('github_token', '')}" placeholder="ghp_...">
                    </div>
                    
                    <div class="form-group full-width">
                        <h3>⚡ Processing Settings</h3>
                    </div>
                    
                    <div class="form-group">
                        <label for="pdf_batch_size">PDF Batch Size</label>
                        <input type="number" id="pdf_batch_size" name="pdf_batch_size" value="{config.get('pdf_batch_size', 100)}" min="10" max="500">
                    </div>
                    
                    <div class="form-group">
                        <label for="max_files_per_commit">Max Files Per Commit</label>
                        <input type="number" id="max_files_per_commit" name="max_files_per_commit" value="{config.get('max_files_per_commit', 100)}" min="10" max="200">
                    </div>
                    
                    <div class="form-group">
                        <label for="max_pdfs_per_run">Max PDFs Per Run</label>
                        <input type="number" id="max_pdfs_per_run" name="max_pdfs_per_run" value="{config.get('max_pdfs_per_run', 50)}" min="1" max="200">
                    </div>
                    
                    <div class="form-group">
                        <label for="image_zoom">Image Zoom Factor</label>
                        <input type="number" id="image_zoom" name="image_zoom" value="{config.get('image_zoom', 4.0)}" min="1.0" max="5.0" step="0.5">
                    </div>
                    
                    <div class="form-group">
                        <label for="image_dpi">Image DPI</label>
                        <input type="number" id="image_dpi" name="image_dpi" value="{config.get('image_dpi', 300)}" min="72" max="600">
                    </div>
                    
                    <div class="form-group">
                        <label for="cron_schedule">Cron Schedule</label>
                        <input type="text" id="cron_schedule" name="cron_schedule" value="{config.get('cron_schedule', '0 */6 * * *')}" placeholder="0 */6 * * *">
                        <small style="color: #666;">Example: 0 */6 * * * (every 6 hours)</small>
                    </div>
                    
                    <div class="form-group">
                        <label for="priority_default">Default Priority</label>
                        <input type="number" id="priority_default" name="priority_default" value="{config.get('priority_default', 5)}" min="1" max="10">
                    </div>
                </div>
                
                <div class="btn-group">
                    <button type="submit" class="btn btn-primary">💾 Save Configuration</button>
                    <button type="button" class="btn btn-secondary" onclick="location.reload()">Reset</button>
                </div>
            </form>
        </div>
        
        <!-- Archives Management Tab -->
        <div id="archives-tab" class="tab-content">
            <h2>Internet Archive Management</h2>
            <p style="color: #666; margin-bottom: 20px;">নতুন আর্কাইভ যোগ করুন বা বিদ্যমান আর্কাইভ ম্যানেজ করুন</p>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>➕ Add New Archive</h3>
                <form id="add-archive-form" onsubmit="addArchive(event)">
                    <div style="display: grid; grid-template-columns: 2fr 3fr 1fr auto; gap: 15px; align-items: end;">
                        <div>
                            <label for="book_name">Book Name *</label>
                            <input type="text" id="book_name" placeholder="তাফসীর ফী যিলালিল কোরআন" required>
                        </div>
                        <div>
                            <label for="archive_url">Archive URL *</label>
                            <input type="text" id="archive_url" placeholder="https://archive.org/details/..." required>
                        </div>
                        <div>
                            <label for="priority">Priority (1-10)</label>
                            <input type="number" id="priority" value="5" min="1" max="10">
                        </div>
                        <div>
                            <button type="submit" class="btn btn-primary">Add Archive</button>
                        </div>
                    </div>
                </form>
            </div>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                <h3>📋 Bulk Import</h3>
                <form id="bulk-import-form">
                    <div class="form-group">
                        <label for="bulk_data">Paste archive URLs (one per line)</label>
                        <textarea id="bulk_data" rows="5" placeholder="Book Name 1|https://archive.org/details/...
Book Name 2|https://archive.org/details/..."></textarea>
                    </div>
                    <button type="submit" class="btn btn-secondary">Import Bulk</button>
                </form>
            </div>
            
            <div class="table-container">
                <h3>📚 Existing Archives</h3>
                <table>
                    <thead>
                        <tr>
                            <th><input type="checkbox"></th>
                            <th>Book Name</th>
                            <th>URL</th>
                            <th>Status</th>
                            <th>Priority</th>
                            <th>Progress</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="archives-table-body">
                        <tr><td colspan="7" style="text-align: center;">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="btn-group">
                <button class="btn btn-success" onclick="processSelected()">▶️ Process Selected</button>
                <button class="btn btn-warning" onclick="resetSelected()">🔄 Reset Failed</button>
                <button class="btn btn-danger" onclick="deleteSelected()">🗑️ Delete Selected</button>
            </div>
        </div>
        
        <!-- GitHub Actions Tab -->
        <div id="github-tab" class="tab-content">
            <h2>GitHub Actions Setup</h2>
            <p style="color: #666; margin-bottom: 20px;">GitHub Actions workflow ফাইল জেনারেট করুন</p>
            
            <div class="alert alert-success">
                <strong>📋 Instructions:</strong><br>
                1. Generate workflow file below<br>
                2. Add the file to your GitHub repository: <code>.github/workflows/process.yml</code><br>
                3. Add the secrets to your GitHub repository<br>
                4. Enable GitHub Actions in your repository settings
            </div>
            
            <button class="btn btn-primary" onclick="generateWorkflow()" style="margin-bottom: 20px;">🔧 Generate Workflow</button>
            
            <div id="workflow-result"></div>
        </div>
        
        <!-- Monitor Tab -->
        <div id="monitor-tab" class="tab-content">
            <h2>System Monitor</h2>
            <p style="color: #666; margin-bottom: 20px;">সিস্টেম স্ট্যাটাস এবং প্রসেসিং মনিটরিং</p>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px;">
                    <h3>MongoDB Status</h3>
                    <div id="mongodb-status">Checking...</div>
                </div>
                
                <div style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; padding: 20px; border-radius: 10px;">
                    <h3>HuggingFace Status</h3>
                    <div id="hf-status">Checking...</div>
                </div>
                
                <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; padding: 20px; border-radius: 10px;">
                    <h3>Active Tasks</h3>
                    <div id="active-tasks">0</div>
                </div>
            </div>
            
            <div style="margin-top: 30px;">
                <h3>Recent Processing Logs</h3>
                <div id="logs-container" style="background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 12px;">
                    Loading logs...
                </div>
            </div>
        </div>
    """ + HTML_FOOTER
    
    return HTMLResponse(content=html_content)

# ============ API Routes ============

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return config_manager.get_config()

@app.post("/api/config")
async def save_config(config: SystemConfig):
    """Save configuration"""
    success = config_manager.save_config(config.dict())
    
    # Re-initialize MongoDB connection with new config
    if config.mongodb_uri:
        config_manager.initialize(config.mongodb_uri, config.mongodb_db)
    
    return {"success": success, "message": "Configuration saved"}

@app.post("/api/test/mongodb")
async def test_mongodb(request: Request):
    """Test MongoDB connection"""
    data = await request.json()
    uri = data.get("uri", "")
    
    if not uri:
        return {"success": False, "message": "URI is required"}
    
    success, message = config_manager.test_mongodb_connection(uri)
    return {"success": success, "message": message}

@app.post("/api/test/hf")
async def test_hf(request: Request):
    """Test HuggingFace connection"""
    data = await request.json()
    token = data.get("token", "")
    
    if not token:
        return {"success": False, "message": "Token is required"}
    
    success, message = config_manager.test_hf_connection(token)
    return {"success": success, "message": message}

@app.get("/api/archives")
async def get_archives():
    """Get all archive items"""
    config = config_manager.get_config()
    
    if not config.get("mongodb_uri"):
        return []
    
    try:
        client = MongoClient(config["mongodb_uri"])
        db = client[config.get("mongodb_db", "tafsir_db")]
        collection = db[config.get("mongodb_collection", "archive_links")]
        
        archives = list(collection.find().sort("created_at", -1).limit(100))
        
        # Convert ObjectId to string
        for archive in archives:
            archive["_id"] = str(archive["_id"])
            if "created_at" in archive:
                archive["created_at"] = archive["created_at"].isoformat()
            if "updated_at" in archive:
                archive["updated_at"] = archive["updated_at"].isoformat()
        
        client.close()
        return archives
        
    except Exception as e:
        print(f"Failed to fetch archives: {e}")
        return []

@app.post("/api/archives")
async def add_archive(item: ArchiveItem):
    """Add a new archive item"""
    config = config_manager.get_config()
    
    if not config.get("mongodb_uri"):
        return {"success": False, "message": "MongoDB not configured"}
    
    try:
        client = MongoClient(config["mongodb_uri"])
        db = client[config.get("mongodb_db", "tafsir_db")]
        collection = db[config.get("mongodb_collection", "archive_links")]
        
        # Generate ID from book name
        import re
        doc_id = re.sub(r'[^\w\-_]', '_', item.book_name.lower().replace(' ', '_'))
        
        document = {
            "_id": doc_id,
            "book_name": item.book_name,
            "url": item.archive_url,
            "status": "pending",
            "priority": item.priority,
            "retry_count": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "metadata": item.metadata or {}
        }
        
        collection.update_one(
            {"_id": doc_id},
            {"$set": document},
            upsert=True
        )
        
        client.close()
        return {"success": True, "message": "Archive added successfully", "id": doc_id}
        
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/api/generate/workflow")
async def generate_workflow():
    """Generate GitHub Actions workflow"""
    config = config_manager.get_config()
    
    workflow = GitHubActionsGenerator.generate_workflow(config)
    instructions = GitHubActionsGenerator.generate_secrets_instructions(config)
    
    return {
        "workflow": workflow,
        "instructions": instructions
    }

@app.get("/api/monitor/status")
async def get_system_status():
    """Get system status"""
    config = config_manager.get_config()
    
    status = {
        "mongodb": {"connected": False, "message": "Not configured"},
        "hf": {"connected": False, "message": "Not configured"},
        "active_tasks": 0
    }
    
    # Check MongoDB
    if config.get("mongodb_uri"):
        success, message = config_manager.test_mongodb_connection(config["mongodb_uri"])
        status["mongodb"] = {"connected": success, "message": message}
        
        if success:
            # Count active tasks
            try:
                client = MongoClient(config["mongodb_uri"])
                db = client[config.get("mongodb_db", "tafsir_db")]
                collection = db[config.get("mongodb_collection", "archive_links")]
                status["active_tasks"] = collection.count_documents({"status": "processing"})
                client.close()
            except:
                pass
    
    # Check HuggingFace
    if config.get("hf_token"):
        success, message = config_manager.test_hf_connection(config["hf_token"])
        status["hf"] = {"connected": success, "message": message}
    
    return status

# ============ Main ============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    # Try to load config from file first
    if CONFIG_FILE.exists():
        config = config_manager.get_config()
        if config.get("mongodb_uri"):
            config_manager.initialize(config["mongodb_uri"], config.get("mongodb_db", "tafsir_config"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
