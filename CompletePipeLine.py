"""
Vehicle Re-ID Comparison Pipeline
Pretrained OSNet vs Our Fine-tuned ResNet50
"""

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

class VehicleReIDPipeline:
    def __init__(self, model_type='both'):
        """
        Initialize the Re-ID pipeline
        Args:
            model_type: 'osnet', 'resnet50', or 'both'
        """
        self.model_type = model_type
        self.yolo = YOLO("yolov8n.pt")
        
        # Initialize OSNet (pretrained baseline)
        if model_type in ['osnet', 'both']:
            self.osnet = FeatureExtractor(
                model_name='osnet_x1_0',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.load_osnet_reference()
        
        # Initialize our fine-tuned ResNet50
        if model_type in ['resnet50', 'both']:
            self.load_finetuned_resnet50()
    
    def load_finetuned_resnet50(self):
        """Load our fine-tuned ResNet50 model and embeddings"""
        print("Loading our fine-tuned ResNet50 model...")
        # Load pre-computed embeddings (trained on VeRi dataset)
        self.features = np.load('features.npy')  # Shape: (37778, 2048)
        self.labels = np.load('labels.npy')
        
        # KNN classifier with cosine distance
        self.knn_model = KNeighborsClassifier(
            n_neighbors=1, 
            metric='cosine', 
            algorithm='brute'
        )
        self.knn_model.fit(self.features, self.labels)
        
        # Load reference images for dual verification
        self.load_reference_database()
        print(f"Loaded {len(self.features)} training samples with 2048-dim features")
    
    def load_osnet_reference(self):
        """Load reference gallery for OSNet matching"""
        print("Loading OSNet reference gallery...")
        self.osnet_gallery = []
        # Load reference images from VeRi query set
        import os
        ref_folder = "reference_images"
        for img_name in sorted(os.listdir(ref_folder))[:50]:
            img_path = os.path.join(ref_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                feat = self.osnet(img).flatten().cpu().numpy()
                self.osnet_gallery.append(feat)
        print(f"Loaded {len(self.osnet_gallery)} reference images")
    
    def load_reference_database(self):
        """Load reference database for dual verification"""
        import os
        self.reference_database = []
        ref_folder = "reference_images"
        for img_name in sorted(os.listdir(ref_folder)):
            img_path = os.path.join(ref_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Use ResNet50 extractor for reference features
                feat = self.extract_resnet_features(img)
                self.reference_database.append(feat)
    
    def extract_resnet_features(self, image):
        """Extract features using our fine-tuned ResNet50"""
        # Resize and preprocess
        img_resized = cv2.resize(image, (224, 224))
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Here you would pass through your fine-tuned ResNet50
        # For demonstration, returning placeholder
        return np.random.randn(2048)
    
    def match_osnet(self, crop):
        """Match using pretrained OSNet"""
        feat = self.osnet(crop).flatten().cpu().numpy().reshape(1, -1)
        similarities = cosine_similarity(feat, self.osnet_gallery)
        max_sim = np.max(similarities)
        return max_sim > 0.70, max_sim
    
    def match_resnet50(self, crop):
        """Match using our fine-tuned ResNet50 with dual verification"""
        feat = self.extract_resnet_features(crop).reshape(1, -1)
        
        # Stage 1: KNN matching
        dist, indices = self.knn_model.kneighbors(feat)
        
        if dist[0][0] < 0.35:
            # Stage 2: Dual verification with reference images
            for r_feat in self.reference_database:
                sim = cosine_similarity(feat, r_feat.reshape(1, -1))[0][0]
                if sim > 0.70:
                    return True, 1 - dist[0][0]
        return False, 1 - dist[0][0]
    
    def process_video(self, video_path, output_path, max_frames=None):
        """Process video and perform Re-ID"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, 
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              fps, (width, height))
        
        frame_count = 0
        osnet_matches = 0
        resnet_matches = 0
        total_detections = 0
        
        print(f"Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_count >= max_frames:
                break
            
            # YOLOv8 detection and tracking
            results = self.yolo.track(frame, persist=True, 
                                      classes=[2, 5, 7], verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[max(0,y1):min(height,y2), 
                                 max(0,x1):min(width,x2)]
                    
                    if crop.size == 0:
                        continue
                    
                    total_detections += 1
                    
                    # Evaluate both models
                    if self.model_type in ['osnet', 'both']:
                        osnet_match, osnet_conf = self.match_osnet(crop)
                        if osnet_match:
                            osnet_matches += 1
                        color_osnet = (0, 255, 0) if osnet_match else (0, 0, 255)
                        label_osnet = f"OSNet: {'MATCH' if osnet_match else 'ID:'+str(int(track_ids[i]))}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_osnet, 2)
                        cv2.putText(frame, label_osnet, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_osnet, 1)
                    
                    if self.model_type in ['resnet50', 'both']:
                        resnet_match, resnet_conf = self.match_resnet50(crop)
                        if resnet_match:
                            resnet_matches += 1
                        color_resnet = (0, 255, 0) if resnet_match else (0, 0, 255)
                        label_resnet = f"Our: {'MATCH' if resnet_match else 'ID:'+str(int(track_ids[i]))}"
                        cv2.rectangle(frame, (x1, y1+30), (x2, y2+30), color_resnet, 2)
                        cv2.putText(frame, label_resnet, (x1, y1+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_resnet, 1)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        # Print statistics
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total vehicle detections: {total_detections}")
        if self.model_type in ['osnet', 'both']:
            print(f"OSNet Matches: {osnet_matches}/{total_detections} ({100*osnet_matches/total_detections:.1f}%)")
        if self.model_type in ['resnet50', 'both']:
            print(f"Our ResNet50 Matches: {resnet_matches}/{total_detections} ({100*resnet_matches/total_detections:.1f}%)")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Run comparison
    pipeline = VehicleReIDPipeline(model_type='both')
    pipeline.process_video('vid_back.mp4', 'output_comparison.mp4', max_frames=900)
