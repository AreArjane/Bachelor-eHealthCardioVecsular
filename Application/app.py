# train_cats_dogs_none.py
import csv, os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT /"Data"/"datasets.csv"
NUM_CLASSES = 3               # cat, dog, none
LABELS = {"cat": 0, "dog": 1, "none": 2}
INV_LABELS = {v: k for k, v in LABELS.items()}
BATCH_SIZE = 12
EPOCHS = 5                    # keep it small; you can increase later
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset that reads from ONE CSV file ---
class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None, strict=False):
        self.items = []
        self.transform = transform
        csv_path = Path(csv_path).resolve()
        base = csv_path.parent

        missing = 0
        badlabel = 0

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):  # header = line 1
                rel = (row.get("path") or "").strip().replace("\\", "/")
                lbl = (row.get("label") or "").strip().lower()
                img_path = (base / rel).resolve()

                if lbl not in LABELS:
                    print(f"[WARN] line {i}: unknown label '{lbl}' (skip)")
                    badlabel += 1
                    continue
                if not img_path.exists():
                    print(f"[WARN] line {i}: not found: {img_path} (skip)")
                    missing += 1
                    continue

                self.items.append((img_path, LABELS[lbl]))

        print(f"[INFO] Loaded {len(self.items)} samples "
              f"(missing: {missing}, bad labels: {badlabel}) from {csv_path}")

        if strict and not self.items:
            raise FileNotFoundError("No valid rows in CSV; check paths/labels.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return (self.transform(img) if self.transform else img), y

# --- Transforms (light augmentation helps when you have ~10 images) ---
train_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Build datasets/loaders ---
full_ds = CsvImageDataset(CSV_PATH, transform=train_tfms)
# If you only have ~10 images, keep most for training
val_size = max(1, len(full_ds) // 5)   # 20% validation (at least 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# For validation, use deterministic transforms
val_ds.dataset.transform = test_tfms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# --- Model: Pretrained ResNet18 → replace final layer to 3 classes ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch}: train loss={running_loss/total:.4f}, acc={100*correct/total:.1f}%")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        print("No validation samples.")
    else:
        print(f"Validation acc: {100*correct/total:.1f}%")

for e in range(1, EPOCHS+1):
    train_one_epoch(e)
    evaluate()

# --- Save the trained model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cat_dog_none_resnet18.pth")
print("Saved weights to models/cat_dog_none_resnet18.pth")

# --- Inference helper ---
def predict_image(image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(prob))
        return INV_LABELS[pred_idx], float(prob[pred_idx])

# Example:
# label, confidence = predict_image("images/some_new_photo.jpg")
# print(label, f"{confidence:.2f}")
