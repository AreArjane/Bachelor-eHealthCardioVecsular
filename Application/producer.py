import torch
from torchvision import models, transforms
from PIL import Image

# Same labels dictionary you used before
LABELS = {0: "cat", 1: "dog", 2: "none"}
NUM_CLASSES = len(LABELS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Rebuild the model (same architecture as training) ---
model = models.resnet18(weights=None)  # no pretrained weights now
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("models/cat_dog_none_resnet18.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return LABELS[pred_idx], float(probs[pred_idx])

# Example usage
label, confidence = predict_image("test.jpg")
print(f"Prediction: {label} ({confidence:.2f})")