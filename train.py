"""
Training script for our fine-tuned ResNet50 model on VeRi dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class TripletLoss(nn.Module):
    """Batch-hard triplet loss with margin"""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, features, labels):
        # Compute pairwise distance matrix
        n = features.size(0)
        dist = torch.cdist(features, features, p=2)
        
        # For each anchor, find hardest positive and negative
        loss = 0
        for i in range(n):
            positive_mask = labels == labels[i]
            negative_mask = labels != labels[i]
            
            if positive_mask.sum() > 1:
                hardest_positive = dist[i, positive_mask].max()
                hardest_negative = dist[i, negative_mask].min()
                loss += torch.relu(hardest_positive - hardest_negative + self.margin)
        
        return loss / n

def train_our_model(train_loader, val_loader, num_epochs=120):
    """
    Train our ResNet50 model on VeRi dataset
    """
    # Build ResNet50 backbone
    import torchvision.models as models
    model = models.resnet50(pretrained=True)
    
    # Replace final layer for Re-ID
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 576)  # 576 vehicle identities
    model = model.cuda()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss functions
    triplet_loss = TripletLoss(margin=0.3)
    ce_loss = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward pass
            features = model(images)
            
            # Combined loss
            loss_triplet = triplet_loss(features, labels)
            loss_ce = ce_loss(features, labels)
            loss = loss_triplet + 0.5 * loss_ce
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            accuracy = validate(model, val_loader)
            print(f"Epoch {epoch+1}: Validation Accuracy = {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'our_finetuned_resnet50_best.pth')
                print(f"Saved best model with accuracy {accuracy:.2f}%")
        
        # Save embeddings periodically
        if (epoch + 1) % 20 == 0:
            save_embeddings(model, train_loader)
    
    return model

def save_embeddings(model, dataloader):
    """Extract and save embeddings for all training images"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.cuda()
            features = model(images).cpu().numpy()
            all_features.extend(features)
            all_labels.extend(labels.numpy())
    
    # Save as numpy arrays
    np.save('features.npy', np.array(all_features))
    np.save('labels.npy', np.array(all_labels))
    print(f"Saved {len(all_features)} embeddings of dimension {all_features[0].shape[0]}")

if __name__ == "__main__":
    # Load VeRi dataset
    from veri_dataset import VeRiDataset
    
    train_dataset = VeRiDataset('veri_data/VeRi', split='train')
    val_dataset = VeRiDataset('veri_data/VeRi', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train our model
    model = train_our_model(train_loader, val_loader)
    print("Training complete! Our fine-tuned model saved.")
