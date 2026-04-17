# extract_test_features.py - Generate features for evaluation
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

# ================= CONFIG =================
VERI_PATH = 'VeRi'  # Path to your VeRi dataset
OUTPUT_DIR = 'evaluation_data'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= MODEL =================
class SpatialEmbeddingNet(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return nn.functional.normalize(x, p=2, dim=1)

# ================= FEATURE EXTRACTOR =================
print("Loading models...")
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone = backbone.to(DEVICE)
backbone.eval()

spatial_model = SpatialEmbeddingNet(input_dim=2048, embedding_dim=256)
if os.path.exists('spatial_reid.pth'):
    spatial_model.load_state_dict(torch.load('spatial_reid.pth', map_location=DEVICE))
    print("Loaded trained spatial model")
spatial_model = spatial_model.to(DEVICE)
spatial_model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract 2048-dim backbone features."""
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = backbone(tensor).squeeze().cpu().numpy()
        return features
    except:
        return None

# ================= EXTRACT TEST FEATURES =================
print("\nExtracting TEST (gallery) features...")
test_path = Path(VERI_PATH) / 'image_test'
test_images = list(test_path.glob('*.jpg'))[:2000]  # Limit for speed

test_features = []
test_labels = []

for img_path in tqdm(test_images):
    features = extract_features(img_path)
    if features is not None:
        test_features.append(features)
        # Extract vehicle ID from filename (e.g., "0001_c001_0001.jpg" -> "0001")
        vehicle_id = img_path.stem.split('_')[0]
        test_labels.append(vehicle_id)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

np.save(os.path.join(OUTPUT_DIR, 'test_features.npy'), test_features)
np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), test_labels)
print(f"✅ Saved {len(test_features)} test features")

# ================= EXTRACT QUERY FEATURES =================
print("\nExtracting QUERY features...")
query_path = Path(VERI_PATH) / 'image_query'
query_images = list(query_path.glob('*.jpg'))[:500]

query_features = []
query_labels = []

for img_path in tqdm(query_images):
    features = extract_features(img_path)
    if features is not None:
        query_features.append(features)
        vehicle_id = img_path.stem.split('_')[0]
        query_labels.append(vehicle_id)

query_features = np.array(query_features)
query_labels = np.array(query_labels)

np.save(os.path.join(OUTPUT_DIR, 'query_features.npy'), query_features)
np.save(os.path.join(OUTPUT_DIR, 'query_labels.npy'), query_labels)
print(f"✅ Saved {len(query_features)} query features")

print("\n" + "="*50)
print("Features saved to:", OUTPUT_DIR)
print("  - test_features.npy")
print("  - test_labels.npy")
print("  - query_features.npy")
print("  - query_labels.npy")
print("="*50)
