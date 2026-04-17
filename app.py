# app.py - Complete Vehicle ReID System with Image Display & Full Metadata
import sys
sys.path.append(r"D:\my_utils")

from my_utils.xml_loader import load_xml
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import faiss
import json
from datetime import datetime
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
import time
from collections import defaultdict
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE_FOLDER = 'database'
VERI_PATH = 'VeRi'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index = None
metadata = {}
veri_metadata = {}
evaluation_results = {}

print("="*60)
print("VEHICLE RE-ID SYSTEM WITH VERI-776 DATASET")
print("="*60)
print(f"Device: {device}")
print(f"VeRi Path: {VERI_PATH}")

# Initialize models
print("\nLoading models...")
print("   Loading ResNet50 feature extractor...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("   Models loaded!")

# Initialize FAISS
dimension = 2048
index_path = os.path.join(DATABASE_FOLDER, 'veri_index.faiss')
metadata_path = os.path.join(DATABASE_FOLDER, 'veri_metadata.json')

if os.path.exists(index_path):
    print(f"\nLoading existing database...")
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"   Loaded {index.ntotal} indexed vehicles")
else:
    print(f"\nCreating new database...")
    index = faiss.IndexFlatIP(dimension)
    metadata = {}
    print(f"   New database created")

def image_to_base64(image_path):
    """Convert image to base64 for web display."""
    try:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
    except:
        pass
    return None

def extract_features(image):
    """Extract 2048-dim features from an image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor).squeeze().cpu().numpy()
    return features / np.linalg.norm(features)

def save_database():
    """Save FAISS index and metadata."""
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_veri_metadata():
    """Load VeRi dataset metadata safely."""
    global veri_metadata
    
    train_xml = os.path.join(VERI_PATH, 'train_label.xml')
    test_xml = os.path.join(VERI_PATH, 'test_label.xml')
    
    if os.path.exists(train_xml):
        tree = load_xml(train_xml)
        root = tree.getroot()
        
        for item in root.findall('Item'):
            img_name = item.get('imageName')
            veri_metadata[img_name] = {
                'vehicle_id': item.get('vehicleID'),
                'camera_id': item.get('cameraID'),
                'color': item.get('colorID'),
                'type': item.get('typeID'),
                'split': 'train'
            }
    
    if os.path.exists(test_xml):
        tree = load_xml(test_xml)
        root = tree.getroot()
        
        for item in root.findall('Item'):
            img_name = item.get('imageName')
            veri_metadata[img_name] = {
                'vehicle_id': item.get('vehicleID'),
                'camera_id': item.get('cameraID'),
                'color': item.get('colorID'),
                'type': item.get('typeID'),
                'split': 'test'
            }
    
    print(f"   Loaded metadata for {len(veri_metadata)} VeRi images")

load_veri_metadata()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle ReID System - Visual Results</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: auto; }
            h1 {
                color: white;
                text-align: center;
                margin-bottom: 20px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .main-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .card { 
                background: white; 
                padding: 25px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .full-width { grid-column: 1 / -1; }
            h2 { 
                color: #333; 
                margin-bottom: 20px;
                border-bottom: 3px solid #2a5298;
                padding-bottom: 10px;
            }
            .upload-area {
                border: 3px dashed #2a5298;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                background: #f8f9ff;
                cursor: pointer;
                margin: 20px 0;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #eef0ff;
                border-color: #1e3c72;
            }
            input[type="file"] { display: none; }
            button { 
                background: #2a5298; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                margin: 5px;
                font-size: 16px;
                transition: all 0.3s;
            }
            button:hover {
                background: #1e3c72;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            button.success { background: #28a745; }
            button.success:hover { background: #218838; }
            button.warning { background: #ffc107; color: #333; }
            button.warning:hover { background: #e0a800; }
            button.danger { background: #dc3545; }
            button.danger:hover { background: #c82333; }
            #query-preview { 
                max-width: 100%; 
                max-height: 300px;
                margin: 20px auto; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                display: block;
            }
            .result-area { 
                margin-top: 20px; 
                padding: 15px; 
                background: #f8f9fa; 
                border-radius: 10px; 
                max-height: 600px;
                overflow-y: auto;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-value {
                font-size: 36px;
                font-weight: bold;
            }
            .stat-label {
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }
            .results-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .vehicle-card {
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .vehicle-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            }
            .vehicle-image {
                width: 100%;
                height: 200px;
                object-fit: cover;
                background: #f0f0f0;
            }
            .vehicle-info {
                padding: 15px;
            }
            .vehicle-id {
                font-weight: bold;
                color: #2a5298;
                margin-bottom: 5px;
            }
            .similarity-bar {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                overflow: hidden;
                margin: 10px 0;
            }
            .similarity-fill {
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                transition: width 0.5s;
            }
            .similarity-text {
                font-weight: bold;
                color: #28a745;
            }
            .metadata-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin-top: 10px;
                font-size: 13px;
            }
            .metadata-item {
                background: #f0f4f8;
                padding: 5px 8px;
                border-radius: 5px;
            }
            .metadata-label {
                color: #666;
                font-size: 11px;
            }
            .metadata-value {
                font-weight: 600;
                color: #333;
            }
            .correct-badge {
                position: absolute;
                top: 10px;
                right: 10px;
                background: #28a745;
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
            }
            .incorrect-badge {
                position: absolute;
                top: 10px;
                right: 10px;
                background: #dc3545;
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
            }
            .image-container {
                position: relative;
            }
            .query-section {
                text-align: center;
                padding: 20px;
                background: #f0f4f8;
                border-radius: 12px;
                margin-bottom: 20px;
            }
            .query-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }
            .tab-container { margin: 20px 0; }
            .tabs {
                display: flex;
                border-bottom: 2px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                background: none;
                border: none;
                color: #666;
                font-size: 16px;
            }
            .tab.active {
                color: #2a5298;
                border-bottom: 3px solid #2a5298;
            }
            .tab-pane { display: none; }
            .tab-pane.active { display: block; }
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vehicle Re-Identification System</h1>
            <p style="text-align: center; color: white; margin-bottom: 20px;">Visual Search with Complete Metadata</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-indexed">0</div>
                    <div class="stat-label">Vehicles Indexed</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #f093fb, #f5576c);">
                    <div class="stat-value" id="veri-train">0</div>
                    <div class="stat-label">VeRi Train Images</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #4facfe, #00f2fe);">
                    <div class="stat-value" id="veri-test">0</div>
                    <div class="stat-label">VeRi Test Images</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #43e97b, #38f9d7);">
                    <div class="stat-value" id="mAP-score">-</div>
                    <div class="stat-label">mAP Score</div>
                </div>
            </div>
            
            <div class="main-grid">
                <div class="card">
                    <h2>Search & Index</h2>
                    
                    <div class="tab-container">
                        <div class="tabs">
                            <button class="tab active" onclick="switchTab('upload')">Upload Image</button>
                            <button class="tab" onclick="switchTab('veri-search')">Search VeRi</button>
                            <button class="tab" onclick="switchTab('batch')">Batch Index</button>
                        </div>
                        
                        <div class="tab-content">
                            <div id="upload-tab" class="tab-pane active">
                                <div class="upload-area" onclick="document.getElementById('image').click()">
                                    <input type="file" id="image" accept="image/*" onchange="previewImage()">
                                    <div id="upload-text">
                                        Click to select an image<br>
                                        <small>or drag and drop</small>
                                    </div>
                                    <img id="preview" style="display:none;">
                                </div>
                                
                                <div style="text-align: center;">
                                    <button onclick="searchImage()">Search Similar</button>
                                    <button onclick="indexImage()" class="success">Index Vehicle</button>
                                </div>
                            </div>
                            
                            <div id="veri-search-tab" class="tab-pane">
                                <h3>Search using VeRi Query Images</h3>
                                <select id="veri-query-select" style="width: 100%; padding: 10px; margin: 10px 0;">
                                    <option value="">Loading query images...</option>
                                </select>
                                <button onclick="searchVeriQuery()" class="warning">Search VeRi Query</button>
                            </div>
                            
                            <div id="batch-tab" class="tab-pane">
                                <h3>Batch Index VeRi Dataset</h3>
                                <label>Number of images to index:</label>
                                <input type="number" id="batch-size" value="1000" min="100" max="10000" style="width: 100%; padding: 10px; margin: 10px 0;">
                                <button onclick="startBatchIndex()" class="success">Start Batch Index</button>
                                <div id="batch-progress" style="margin-top: 15px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Evaluation Metrics</h2>
                    
                    <button onclick="runEvaluation()" class="warning" style="width: 100%; margin-bottom: 20px;">
                        Run VeRi-776 Evaluation
                    </button>
                    
                    <div id="evaluation-results">
                        <p style="color: #999; text-align: center;">Click "Run Evaluation" to calculate metrics</p>
                    </div>
                    
                    <h3 style="margin-top: 30px;">Database Info</h3>
                    <div id="database-info">
                        <p>Loading database info...</p>
                    </div>
                    
                    <button onclick="clearDatabase()" class="danger" style="width: 100%; margin-top: 20px;">
                        Clear Database
                    </button>
                </div>
            </div>
            
            <div class="card full-width" style="margin-top: 20px;">
                <h2>Search Results</h2>
                <div id="search-results">
                    <p style="color: #999; text-align: center;">Upload an image to see similar vehicles with images and metadata</p>
                </div>
            </div>
        </div>
        
        <script>
            let currentImageFile = null;
            
            function switchTab(tabName) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                
                event.target.classList.add('active');
                document.getElementById(tabName + '-tab').classList.add('active');
                
                if (tabName === 'veri-search') {
                    loadVeriQueries();
                }
            }
            
            function previewImage() {
                const file = document.getElementById('image').files[0];
                if (file) {
                    currentImageFile = file;
                    const reader = new FileReader();
                    reader.onload = e => {
                        const img = document.getElementById('preview');
                        img.src = e.target.result;
                        img.style.display = 'block';
                        document.getElementById('upload-text').style.display = 'none';
                    };
                    reader.readAsDataURL(file);
                }
            }
            
            async function searchImage() {
                if (!currentImageFile) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', currentImageFile);
                
                document.getElementById('search-results').innerHTML = '<div class="loading">Searching database...</div>';
                
                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.success && data.results.length > 0) {
                        let html = `
                            <div class="query-section">
                                <div class="query-title">Query Image</div>
                                <img src="${data.query_image}" alt="Query" style="max-width: 300px; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                                <p style="margin-top: 10px; color: #666;">Found ${data.results.length} matches in ${data.search_time_ms}ms</p>
                            </div>
                        `;
                        
                        html += '<div class="results-grid">';
                        
                        data.results.forEach((r, i) => {
                            const similarity = (r.similarity * 100).toFixed(1);
                            const isCorrect = r.metadata.veri_id === data.query_vehicle_id;
                            
                            html += `
                                <div class="vehicle-card">
                                    <div class="image-container">
                                        <img src="${r.image}" alt="Match ${i+1}" class="vehicle-image">
                                        ${isCorrect ? '<span class="correct-badge">✓ Correct Match</span>' : '<span class="incorrect-badge">✗ Different</span>'}
                                    </div>
                                    <div class="vehicle-info">
                                        <div class="vehicle-id">Match #${i+1}</div>
                                        <div class="similarity-bar">
                                            <div class="similarity-fill" style="width: ${similarity}%;"></div>
                                        </div>
                                        <div><span class="similarity-text">${similarity}% Similar</span></div>
                                        
                                        <div class="metadata-grid">
                                            <div class="metadata-item">
                                                <div class="metadata-label">Vehicle ID</div>
                                                <div class="metadata-value">${r.metadata.veri_id || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Camera ID</div>
                                                <div class="metadata-value">${r.metadata.camera_id || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Color</div>
                                                <div class="metadata-value">${r.metadata.color || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Type</div>
                                                <div class="metadata-value">${r.metadata.type || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Source</div>
                                                <div class="metadata-value">${r.metadata.source || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Filename</div>
                                                <div class="metadata-value">${r.metadata.filename || 'N/A'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                        
                        html += '</div>';
                        document.getElementById('search-results').innerHTML = html;
                    } else {
                        document.getElementById('search-results').innerHTML = '<p style="color: #999; text-align: center;">No matches found</p>';
                    }
                    
                    updateStats();
                } catch (error) {
                    document.getElementById('search-results').innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
            
            async function indexImage() {
                if (!currentImageFile) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', currentImageFile);
                
                document.getElementById('search-results').innerHTML = '<div class="loading">Indexing vehicle...</div>';
                
                try {
                    const response = await fetch('/index', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('search-results').innerHTML = 
                            `<div class="query-section">
                                <h3>Vehicle Indexed Successfully!</h3>
                                <p>Vehicle ID: ${data.vehicle_id}</p>
                                <p>Total vehicles in database: ${data.total_indexed}</p>
                            </div>`;
                        updateStats();
                    } else {
                        document.getElementById('search-results').innerHTML = 
                            `<p style="color: red;">${data.message}</p>`;
                    }
                } catch (error) {
                    document.getElementById('search-results').innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
            
            async function searchVeriQuery() {
                const queryImage = document.getElementById('veri-query-select').value;
                if (!queryImage) {
                    alert('Please select a query image');
                    return;
                }
                
                document.getElementById('search-results').innerHTML = '<div class="loading">Searching VeRi database...</div>';
                
                try {
                    const response = await fetch(`/veri/search?query=${encodeURIComponent(queryImage)}`);
                    const data = await response.json();
                    
                    if (data.success && data.results.length > 0) {
                        let html = `
                            <div class="query-section">
                                <div class="query-title">Query: ${queryImage}</div>
                                <img src="${data.query_image}" alt="Query" style="max-width: 300px; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                                <p style="margin-top: 10px;">Ground Truth ID: <strong>${data.ground_truth || 'Unknown'}</strong></p>
                                <p>Found ${data.results.length} matches</p>
                            </div>
                        `;
                        
                        html += '<div class="results-grid">';
                        
                        data.results.forEach((r, i) => {
                            const similarity = (r.similarity * 100).toFixed(1);
                            const isCorrect = r.vehicle_id === data.ground_truth;
                            
                            html += `
                                <div class="vehicle-card">
                                    <div class="image-container">
                                        <img src="${r.image}" alt="Match ${i+1}" class="vehicle-image">
                                        ${isCorrect ? '<span class="correct-badge">✓ Correct</span>' : '<span class="incorrect-badge">✗ Different</span>'}
                                    </div>
                                    <div class="vehicle-info">
                                        <div class="vehicle-id">Rank #${i+1}</div>
                                        <div class="similarity-bar">
                                            <div class="similarity-fill" style="width: ${similarity}%;"></div>
                                        </div>
                                        <div><span class="similarity-text">${similarity}% Similar</span></div>
                                        
                                        <div class="metadata-grid">
                                            <div class="metadata-item">
                                                <div class="metadata-label">Vehicle ID</div>
                                                <div class="metadata-value">${r.vehicle_id || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Camera ID</div>
                                                <div class="metadata-value">${r.camera_id || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Color</div>
                                                <div class="metadata-value">${r.color || 'N/A'}</div>
                                            </div>
                                            <div class="metadata-item">
                                                <div class="metadata-label">Type</div>
                                                <div class="metadata-value">${r.type || 'N/A'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                        
                        html += '</div>';
                        document.getElementById('search-results').innerHTML = html;
                    } else {
                        document.getElementById('search-results').innerHTML = '<p style="color: #999;">No matches found</p>';
                    }
                } catch (error) {
                    document.getElementById('search-results').innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
            
            async function loadVeriQueries() {
                try {
                    const response = await fetch('/veri/queries');
                    const data = await response.json();
                    
                    let options = '<option value="">Select a query image...</option>';
                    data.queries.forEach(q => {
                        options += `<option value="${q}">${q}</option>`;
                    });
                    document.getElementById('veri-query-select').innerHTML = options;
                } catch (error) {
                    console.error('Failed to load queries:', error);
                }
            }
            
            async function startBatchIndex() {
                const batchSize = document.getElementById('batch-size').value;
                
                document.getElementById('batch-progress').innerHTML = 
                    `<div style="text-align: center; padding: 20px;">
                        <p>Indexing ${batchSize} images from VeRi dataset...</p>
                        <p>This may take a few minutes...</p>
                    </div>`;
                
                try {
                    const response = await fetch('/veri/index-batch', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({limit: parseInt(batchSize)})
                    });
                    const data = await response.json();
                    
                    document.getElementById('batch-progress').innerHTML = 
                        `<div style="text-align: center; padding: 20px; background: #d4edda; border-radius: 10px;">
                            <h3 style="color: #155724;">Batch Index Complete!</h3>
                            <p>Indexed ${data.indexed} new vehicles</p>
                            <p>Total in database: ${data.total_in_database}</p>
                        </div>`;
                    
                    updateStats();
                } catch (error) {
                    document.getElementById('batch-progress').innerHTML = 
                        '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
            
            async function runEvaluation() {
                document.getElementById('evaluation-results').innerHTML = '<div class="loading">Running evaluation...</div>';
                
                try {
                    const response = await fetch('/evaluate', { method: 'POST' });
                    const data = await response.json();
                    
                    let html = `<h3>Evaluation Results</h3>`;
                    html += `<table style="width: 100%; border-collapse: collapse;">
                        <tr style="background: #2a5298; color: white;">
                            <th style="padding: 10px;">Metric</th>
                            <th style="padding: 10px;">Score</th>
                        </tr>
                        <tr><td style="padding: 10px;">mAP</td><td style="padding: 10px;"><strong>${data.mAP.toFixed(2)}%</strong></td></tr>
                        <tr><td style="padding: 10px;">Rank-1</td><td style="padding: 10px;">${data.rank1.toFixed(2)}%</td></tr>
                        <tr><td style="padding: 10px;">Rank-5</td><td style="padding: 10px;">${data.rank5.toFixed(2)}%</td></tr>
                        <tr><td style="padding: 10px;">Rank-10</td><td style="padding: 10px;">${data.rank10.toFixed(2)}%</td></tr>
                        <tr><td style="padding: 10px;">Queries Evaluated</td><td style="padding: 10px;">${data.total_queries}</td></tr>
                    </table>`;
                    
                    document.getElementById('evaluation-results').innerHTML = html;
                    document.getElementById('mAP-score').textContent = data.mAP.toFixed(1) + '%';
                } catch (error) {
                    document.getElementById('evaluation-results').innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                }
            }
            
            async function updateStats() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    
                    document.getElementById('total-indexed').textContent = data.total_vehicles;
                    document.getElementById('veri-train').textContent = data.veri_train_count || 0;
                    document.getElementById('veri-test').textContent = data.veri_test_count || 0;
                    
                    let dbInfo = `<p><strong>Total indexed:</strong> ${data.total_vehicles}</p>`;
                    dbInfo += `<p><strong>VeRi train images:</strong> ${data.veri_train_count || 0}</p>`;
                    dbInfo += `<p><strong>VeRi test images:</strong> ${data.veri_test_count || 0}</p>`;
                    dbInfo += `<p><strong>Database size:</strong> ${data.database_size_mb.toFixed(2)} MB</p>`;
                    document.getElementById('database-info').innerHTML = dbInfo;
                } catch (error) {
                    console.error('Failed to update stats:', error);
                }
            }
            
            async function clearDatabase() {
                if (!confirm('Are you sure you want to clear the entire database?')) return;
                
                try {
                    await fetch('/database/clear', {method: 'POST'});
                    updateStats();
                    document.getElementById('search-results').innerHTML = '<p style="color: #999;">Database cleared</p>';
                } catch (error) {
                    alert('Failed to clear database');
                }
            }
            
            // Initialize
            updateStats();
        </script>
    </body>
    </html>
    '''

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    if index.ntotal == 0:
        return jsonify({
            'success': False,
            'message': 'Database is empty. Index some vehicles first!'
        })
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"query_{uuid.uuid4()}.jpg")
    file.save(filepath)
    
    start_time = time.time()
    
    # Extract features
    features = extract_features(filepath)
    
    # Search
    distances, indices = index.search(features.reshape(1, -1).astype('float32'), min(10, index.ntotal))
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx != -1:
            vehicle_id = list(metadata.keys())[idx]
            meta = metadata[vehicle_id].copy()
            
            # Get image path and convert to base64
            img_path = None
            if meta.get('source') == 'veri_train':
                img_path = os.path.join(VERI_PATH, 'image_train', meta.get('filename', ''))
            elif meta.get('source') == 'veri_test':
                img_path = os.path.join(VERI_PATH, 'image_test', meta.get('filename', ''))
            
            img_base64 = None
            if img_path and os.path.exists(img_path):
                img_base64 = image_to_base64(img_path)
            
            results.append({
                'id': vehicle_id,
                'similarity': float(score),
                'metadata': meta,
                'image': f"data:image/jpeg;base64,{img_base64}" if img_base64 else None
            })
    
    search_time = (time.time() - start_time) * 1000
    
    # Query image base64
    query_base64 = image_to_base64(filepath)
    
    return jsonify({
        'success': True,
        'query_image': f"data:image/jpeg;base64,{query_base64}",
        'results': results,
        'search_time_ms': round(search_time, 2)
    })

@app.route('/index', methods=['POST'])
def index_vehicle():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"indexed_{uuid.uuid4()}.jpg")
    file.save(filepath)
    
    features = extract_features(filepath)
    vehicle_id = str(uuid.uuid4())
    index.add(features.reshape(1, -1).astype('float32'))
    metadata[vehicle_id] = {
        'filename': filename,
        'filepath': filepath,
        'timestamp': datetime.now().isoformat(),
        'source': 'upload'
    }
    
    save_database()
    
    return jsonify({
        'success': True,
        'vehicle_id': vehicle_id[:8],
        'total_indexed': index.ntotal
    })

@app.route('/veri/index-batch', methods=['POST'])
def index_veri_batch():
    data = request.json
    limit = data.get('limit', 1000)
    
    train_path = os.path.join(VERI_PATH, 'image_train')
    if not os.path.exists(train_path):
        return jsonify({'error': 'VeRi training images not found'}), 404
    
    image_files = list(Path(train_path).glob('*.jpg'))[:limit]
    indexed_count = 0
    
    for img_path in image_files:
        try:
            already_indexed = any(m.get('filename') == img_path.name for m in metadata.values())
            if already_indexed:
                continue
            
            features = extract_features(str(img_path))
            if features is not None:
                vehicle_id = str(uuid.uuid4())
                index.add(features.reshape(1, -1).astype('float32'))
                
                meta = veri_metadata.get(img_path.name, {})
                metadata[vehicle_id] = {
                    'filename': img_path.name,
                    'veri_id': meta.get('vehicle_id', 'unknown'),
                    'camera_id': meta.get('camera_id', 'unknown'),
                    'color': meta.get('color', 'unknown'),
                    'type': meta.get('type', 'unknown'),
                    'source': 'veri_train',
                    'indexed_at': datetime.now().isoformat()
                }
                indexed_count += 1
        except Exception as e:
            print(f"Error indexing {img_path}: {e}")
    
    if indexed_count > 0:
        save_database()
    
    return jsonify({
        'success': True,
        'indexed': indexed_count,
        'total_in_database': index.ntotal
    })

@app.route('/veri/queries', methods=['GET'])
def get_veri_queries():
    query_path = os.path.join(VERI_PATH, 'image_query')
    if not os.path.exists(query_path):
        return jsonify({'queries': []})
    
    queries = sorted([f.name for f in Path(query_path).glob('*.jpg')])[:100]
    return jsonify({'queries': queries})

@app.route('/veri/search', methods=['GET'])
def search_veri_query():
    query_name = request.args.get('query')
    if not query_name:
        return jsonify({'error': 'No query specified'}), 400
    
    query_path = os.path.join(VERI_PATH, 'image_query', query_name)
    if not os.path.exists(query_path):
        return jsonify({'error': 'Query image not found'}), 404
    
    features = extract_features(query_path)
    distances, indices = index.search(features.reshape(1, -1).astype('float32'), min(20, index.ntotal))
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx != -1:
            vehicle_id = list(metadata.keys())[idx]
            meta = metadata[vehicle_id].copy()
            
            # Get image
            img_path = None
            if meta.get('source') == 'veri_train':
                img_path = os.path.join(VERI_PATH, 'image_train', meta.get('filename', ''))
            elif meta.get('source') == 'veri_test':
                img_path = os.path.join(VERI_PATH, 'image_test', meta.get('filename', ''))
            
            img_base64 = None
            if img_path and os.path.exists(img_path):
                img_base64 = image_to_base64(img_path)
            
            results.append({
                'id': vehicle_id,
                'similarity': float(score),
                'vehicle_id': meta.get('veri_id'),
                'camera_id': meta.get('camera_id'),
                'color': meta.get('color'),
                'type': meta.get('type'),
                'filename': meta.get('filename'),
                'image': f"data:image/jpeg;base64,{img_base64}" if img_base64 else None
            })
    
    ground_truth = None
    if query_name in veri_metadata:
        ground_truth = veri_metadata[query_name].get('vehicle_id')
    
    query_base64 = image_to_base64(query_path)
    
    return jsonify({
        'success': True,
        'query_image': f"data:image/jpeg;base64,{query_base64}",
        'results': results,
        'ground_truth': ground_truth
    })

@app.route('/evaluate', methods=['POST'])
def evaluate():
    query_path = os.path.join(VERI_PATH, 'image_query')
    test_path = os.path.join(VERI_PATH, 'image_test')
    
    if not os.path.exists(query_path) or not os.path.exists(test_path):
        return jsonify({'error': 'VeRi dataset not found'}), 404
    
    test_images = list(Path(test_path).glob('*.jpg'))[:500]
    gallery_features = []
    gallery_ids = []
    
    for img_path in test_images:
        features = extract_features(str(img_path))
        if features is not None:
            gallery_features.append(features)
            gallery_ids.append(img_path.name)
    
    if not gallery_features:
        return jsonify({'error': 'No gallery features extracted'}), 500
    
    gallery_index = faiss.IndexFlatIP(dimension)
    gallery_index.add(np.array(gallery_features).astype('float32'))
    
    query_images = list(Path(query_path).glob('*.jpg'))[:100]
    all_ap = []
    cmc_scores = defaultdict(int)
    
    for q_img in query_images:
        q_features = extract_features(str(q_img))
        if q_features is None:
            continue
        
        distances, indices = gallery_index.search(q_features.reshape(1, -1).astype('float32'), 50)
        
        q_meta = veri_metadata.get(q_img.name, {})
        q_vehicle_id = q_meta.get('vehicle_id')
        
        if q_vehicle_id:
            gt_indices = []
            for i, g_id in enumerate(gallery_ids):
                g_meta = veri_metadata.get(g_id, {})
                if g_meta.get('vehicle_id') == q_vehicle_id:
                    gt_indices.append(i)
            
            if gt_indices:
                ap = 0
                correct = 0
                for i, idx in enumerate(indices[0]):
                    if idx in gt_indices:
                        correct += 1
                        ap += correct / (i + 1)
                ap = ap / len(gt_indices)
                all_ap.append(ap)
                
                for k in [1, 5, 10]:
                    if any(idx in gt_indices for idx in indices[0][:k]):
                        cmc_scores[k] += 1
    
    total_queries = len(all_ap)
    mAP = np.mean(all_ap) * 100 if all_ap else 0
    
    results = {
        'mAP': mAP,
        'rank1': (cmc_scores[1] / total_queries * 100) if total_queries > 0 else 0,
        'rank5': (cmc_scores[5] / total_queries * 100) if total_queries > 0 else 0,
        'rank10': (cmc_scores[10] / total_queries * 100) if total_queries > 0 else 0,
        'total_queries': total_queries
    }
    
    return jsonify(results)

@app.route('/stats', methods=['GET'])
def get_stats():
    veri_train = sum(1 for m in metadata.values() if m.get('source') == 'veri_train')
    veri_test = sum(1 for m in metadata.values() if m.get('source') == 'veri_test')
    
    db_size = os.path.getsize(index_path) / 1024 / 1024 if os.path.exists(index_path) else 0
    
    return jsonify({
        'total_vehicles': index.ntotal if index else 0,
        'veri_train_count': veri_train,
        'veri_test_count': veri_test,
        'database_size_mb': round(db_size, 2)
    })

@app.route('/database/clear', methods=['POST'])
def clear_database():
    global index, metadata
    index = faiss.IndexFlatIP(dimension)
    metadata = {}
    
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    return jsonify({'success': True, 'message': 'Database cleared'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SYSTEM READY!")
    print("="*60)
    print(f"Open: http://localhost:5000")
    print(f"Database size: {index.ntotal if index else 0} vehicles")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)