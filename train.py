# train.py - Fixed for nested XML structure
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Configuration
class Config:
    VERI_PATH = 'VeRi'
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 256
    MARGIN = 0.5
    NUM_WORKERS = 0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_PATH = 'models'
    
os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

def parse_xml_file(xml_path):
    """Parse XML file - FIXED for nested Items structure."""
    if not os.path.exists(xml_path):
        print(f"   WARNING: {xml_path} not found!")
        return []
    
    items = []
    
    # Read file with proper encoding
    with open(xml_path, 'rb') as f:
        content = f.read()
    
    # The XML says encoding="gb2312" but actually works with utf-8
    for encoding in ['utf-8', 'gb2312', 'gbk', 'iso-8859-1']:
        try:
            text = content.decode(encoding)
            if text.startswith('\ufeff'):
                text = text[1:]
            root = ET.fromstring(text)
            
            # Navigate to Items -> Item (nested structure)
            items_element = root.find('Items')
            if items_element is not None:
                for item in items_element.findall('Item'):
                    items.append(item)
            else:
                # Fallback: look for Item anywhere
                items = root.findall('.//Item')
            
            if items:
                print(f"   Found {len(items)} items in XML")
                return items
        except Exception as e:
            continue
    
    print(f"   WARNING: Could not parse {xml_path}")
    return []

class VeriDataset(Dataset):
    """VeRi Dataset - FIXED for nested XML structure."""
    def __init__(self, root_path, split='train', transform=None):
        self.root_path = Path(root_path)
        self.transform = transform
        self.split = split
        
        print(f"\n📂 Loading {split} dataset...")
        
        # Load labels from XML
        self.labels = {}
        xml_path = self.root_path / f'{split}_label.xml'
        
        print(f"   Reading XML: {xml_path}")
        items = parse_xml_file(xml_path)
        
        for item in items:
            img_name = item.get('imageName')
            if img_name:
                self.labels[img_name] = {
                    'vehicle_id': item.get('vehicleID'),
                    'camera_id': item.get('cameraID'),
                    'color': item.get('colorID'),
                    'type': item.get('typeID')
                }
        
        print(f"   Loaded {len(self.labels)} labels from XML")
        
        # Get image paths
        image_dir = self.root_path / f'image_{split}'
        if image_dir.exists():
            self.image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        else:
            self.image_paths = []
        
        print(f"   Found {len(self.image_paths)} images in folder")
        
        # Filter to images with labels
        self.valid_paths = []
        for img_path in self.image_paths:
            if img_path.name in self.labels:
                self.valid_paths.append(img_path)
        
        print(f"   Images with labels: {len(self.valid_paths)}")
        
        # Group by vehicle ID
        self.vehicle_to_images = defaultdict(list)
        for img_path in self.valid_paths:
            vid = self.labels[img_path.name]['vehicle_id']
            if vid:
                self.vehicle_to_images[vid].append(img_path)
        
        self.vehicle_ids = list(self.vehicle_to_images.keys())
        print(f"   Unique vehicles: {len(self.vehicle_ids)}")
        
        if len(self.valid_paths) == 0:
            raise ValueError(f"No valid images found for {split} split!")
    
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        anchor_path = self.valid_paths[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')
        anchor_vid = self.labels[anchor_path.name]['vehicle_id']
        
        # Positive (same vehicle)
        positive_paths = [p for p in self.vehicle_to_images[anchor_vid] if p != anchor_path]
        if positive_paths:
            positive_path = random.choice(positive_paths)
        else:
            positive_path = anchor_path
        positive_img = Image.open(positive_path).convert('RGB')
        
        # Negative (different vehicle)
        negative_vids = [v for v in self.vehicle_ids if v != anchor_vid]
        if negative_vids:
            negative_vid = random.choice(negative_vids)
            negative_path = random.choice(self.vehicle_to_images[negative_vid])
            negative_img = Image.open(negative_path).convert('RGB')
        else:
            negative_img = anchor_img
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img, anchor_vid

class VehicleReIDModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

def train_model():
    print("="*60)
    print("TRAINING VEHICLE REID MODEL ON VERI-776")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    
    # Data transforms
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(224),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading training data...")
    train_dataset = VeriDataset(Config.VERI_PATH, 'train', train_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"\n✅ Training samples: {len(train_dataset)}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Model
    model = VehicleReIDModel(Config.EMBEDDING_DIM).to(Config.DEVICE)
    criterion = TripletLoss(Config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    history = {'loss': [], 'lr': []}
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for anchor, positive, negative, _ in pbar:
            anchor = anchor.to(Config.DEVICE)
            positive = positive.to(Config.DEVICE)
            negative = negative.to(Config.DEVICE)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / batch_count
        history['loss'].append(avg_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{Config.MODEL_SAVE_PATH}/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"{Config.MODEL_SAVE_PATH}/vehicle_reid_final.pth")
    print(f"\n✅ Model saved to {Config.MODEL_SAVE_PATH}/vehicle_reid_final.pth")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['loss'], 'b-', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(history['lr'], 'r-', linewidth=2)
    axes[1].set_title('Learning Rate')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('LR')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_SAVE_PATH}/training_history.png", dpi=150)
    plt.show()
    
    return model

def evaluate_model(model):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    model.eval()
    
    test_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading test data...")
    test_dataset = VeriDataset(Config.VERI_PATH, 'test', test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✅ Test samples: {len(test_dataset)}")
    
    all_embeddings = []
    all_labels = []
    
    print("\nExtracting embeddings...")
    with torch.no_grad():
        for images, _, _, labels in tqdm(test_loader):
            images = images.to(Config.DEVICE)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels)
    
    all_embeddings = np.vstack(all_embeddings)
    print(f"Extracted {len(all_embeddings)} embeddings")
    
    # Sample evaluation
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_samples = min(1000, len(all_labels))
    indices = np.random.choice(len(all_labels), n_samples, replace=False)
    
    sampled_embeddings = all_embeddings[indices]
    sampled_labels = [all_labels[i] for i in indices]
    
    similarity_matrix = cosine_similarity(sampled_embeddings)
    
    y_true = []
    y_pred = []
    threshold = 0.7
    
    for i in range(len(sampled_labels)):
        for j in range(i+1, len(sampled_labels)):
            y_true.append(1 if sampled_labels[i] == sampled_labels[j] else 0)
            y_pred.append(1 if similarity_matrix[i, j] > threshold else 0)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'threshold': threshold,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{Config.MODEL_SAVE_PATH}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == '__main__':
    print("\n" + "="*60)
    print("VEHICLE RE-IDENTIFICATION TRAINING PIPELINE")
    print("="*60)
    
    if not os.path.exists(Config.VERI_PATH):
        print(f"\n❌ ERROR: VeRi dataset not found at '{Config.VERI_PATH}'")
        exit(1)
    
    print("\n[1/2] Training model...")
    model = train_model()
    
    print("\n[2/2] Evaluating model...")
    metrics = evaluate_model(model)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)