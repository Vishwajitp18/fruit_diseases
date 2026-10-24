import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path

VALIDATOR_CLASSES = [
    "apple_leaf",
    "mango_leaf",
    "banana_leaf",
    "other_leaf_or_flower",
    "non_plant",
]

def main():
    torch.manual_seed(42)
    random.seed(42)

    DATA_DIR = Path("fruitdisease/backend/data/validator_dataset")
    OUTPUT_MODEL_PATH = Path("fruitdisease/backend/app/models/leaf_validator.pth")

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {DEVICE}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=train_transform)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(VALIDATOR_CLASSES)}
    remapped_samples = []
    for path, old_idx in full_dataset.samples:
        class_name = full_dataset.classes[old_idx]
        new_idx = class_to_idx[class_name]
        remapped_samples.append((path, new_idx))
    full_dataset.samples = remapped_samples
    full_dataset.targets = [s[1] for s in remapped_samples]
    full_dataset.class_to_idx = class_to_idx
    full_dataset.classes = VALIDATOR_CLASSES

    print(f"Mapped classes: {full_dataset.class_to_idx}")
    print(f"Total images: {len(full_dataset)}")

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(VALIDATOR_CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    epochs = 12
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

    print(f"\nTraining Complete! Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved validator weights to: {OUTPUT_MODEL_PATH}")

if __name__ == '__main__':
    main()

