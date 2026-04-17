# evaluate_veri.py - VeRi-776 Benchmark Evaluation (FIXED)

import sys
sys.path.append(r"D:\my_utils")

from my_utils.xml_loader import load_xml

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import faiss
import json
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict


class VeRiEvaluator:
    def __init__(self, veri_path="./VeRi"):
        self.veri_path = Path(veri_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔄 Using device: {self.device}")
        print("🔄 Loading feature extractor (ResNet50)...")

        # Model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Load labels (FIXED)
        self.train_labels = self._load_labels('train_label.xml')
        self.test_labels = self._load_labels('test_label.xml')

        print(f"✅ Loaded {len(self.train_labels)} training labels")
        print(f"✅ Loaded {len(self.test_labels)} test labels")

    def _load_labels(self, xml_file):
        """Load XML labels safely (handles UTF-16 encoding)."""
        labels = {}
        xml_path = self.veri_path / xml_file

        if not xml_path.exists():
            print(f"⚠️ {xml_file} not found")
            return labels

        # ✅ FIXED HERE
        tree = load_xml(str(xml_path))
        root = tree.getroot()

        for item in root.findall('Item'):
            img_name = item.get('imageName')
            labels[img_name] = {
                'vehicle_id': item.get('vehicleID'),
                'camera_id': item.get('cameraID'),
                'color': item.get('colorID'),
                'type': item.get('typeID')
            }

        return labels

    def extract_features(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(tensor).squeeze().cpu().numpy()

            return features / np.linalg.norm(features)

        except Exception:
            return None

    def build_gallery_index(self):
        print("\n📦 Building gallery index...")

        test_images = list((self.veri_path / 'image_test').glob('*.jpg'))
        features_list = []
        self.gallery_ids = []

        for img_path in tqdm(test_images[:2000]):
            features = self.extract_features(img_path)

            if features is not None:
                features_list.append(features)
                self.gallery_ids.append(img_path.name)

        if features_list:
            self.index = faiss.IndexFlatIP(2048)
            self.index.add(np.array(features_list).astype('float32'))
            print(f"✅ Indexed {self.index.ntotal} images")
        else:
            print("❌ No features extracted")

    def load_ground_truth(self):
        gt_path = self.veri_path / 'gt_index.txt'

        if not gt_path.exists():
            print("⚠️ gt_index.txt not found")
            return

        with open(gt_path, 'r') as f:
            lines = f.readlines()

        self.gt_matches = {}
        for i, line in enumerate(lines):
            self.gt_matches[i] = [int(x) for x in line.split()]

        print(f"✅ Loaded ground truth for {len(self.gt_matches)} queries")

    def evaluate_queries(self, top_k=50):
        print("\n🔍 Evaluating queries...")

        query_images = list((self.veri_path / 'image_query').glob('*.jpg'))

        all_ap = []
        cmc = defaultdict(int)
        total = 0

        for i, img_path in enumerate(tqdm(query_images[:500])):
            features = self.extract_features(img_path)
            if features is None or i not in self.gt_matches:
                continue

            distances, indices = self.index.search(
                features.reshape(1, -1).astype('float32'),
                top_k
            )

            gt = set(self.gt_matches[i])

            correct = 0
            ap = 0

            for rank, idx in enumerate(indices[0]):
                if idx in gt:
                    correct += 1
                    ap += correct / (rank + 1)

            ap = ap / len(gt) if len(gt) > 0 else 0
            all_ap.append(ap)

            for k in [1, 5, 10]:
                if any(idx in gt for idx in indices[0][:k]):
                    cmc[k] += 1

            total += 1

        mAP = np.mean(all_ap) * 100 if all_ap else 0

        print("\n📊 RESULTS")
        print("=" * 50)
        print(f"mAP: {mAP:.2f}%")
        print(f"Rank-1: {cmc[1]/total*100:.2f}%")
        print(f"Rank-5: {cmc[5]/total*100:.2f}%")
        print(f"Rank-10: {cmc[10]/total*100:.2f}%")
        print("=" * 50)

        return {
            "mAP": mAP,
            "Rank-1": cmc[1]/total*100,
            "Rank-5": cmc[5]/total*100,
            "Rank-10": cmc[10]/total*100
        }

    def run_full_evaluation(self):
        print("\n🚀 Starting Evaluation...")

        start = time.time()

        self.build_gallery_index()
        self.load_ground_truth()
        results = self.evaluate_queries()

        print(f"\n⏱️ Time: {time.time() - start:.1f}s")

        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    evaluator = VeRiEvaluator(veri_path="./VeRi")
    evaluator.run_full_evaluation()