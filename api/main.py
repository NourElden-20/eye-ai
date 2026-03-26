from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import cv2
import numpy as np
import timm

app = FastAPI()

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

# ==================== Diabetes Model ====================
diabetes_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
diabetes_model.classifier[1] = nn.Linear(diabetes_model.classifier[1].in_features, 5)
diabetes_model.load_state_dict(torch.load("models/best_diabetes_model.pth", map_location=DEVICE))
diabetes_model = diabetes_model.to(DEVICE)
diabetes_model.eval()
print("✅ Diabetes Model loaded!")

# ==================== Anemia Model ====================
anemia_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
anemia_model.load_state_dict(torch.load("models/best_anemia_model.pth", map_location=DEVICE))
anemia_model = anemia_model.to(DEVICE)
anemia_model.eval()
print("✅ Anemia Model loaded!")

# ==================== Transform ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==================== Classes ====================
DIABETES_CLASSES = {
    0: "No Diabetic Retinopathy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

DIABETES_SEVERITY = {
    0: "normal",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "severe"
}

ANEMIA_CLASSES = {
    0: "Anemic",
    1: "Non-anemic"
}

# ==================== Preprocessing ====================
def preprocess_eye_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    img = cv2.addWeighted(img, 4,
                          cv2.GaussianBlur(img, (0, 0), 30), -4, 128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# ==================== Endpoints ====================
@app.get("/")
def root():
    return {"message": "Eye Diagnosis API is running!"}

@app.post("/predict/diabetes")
async def predict_diabetes(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = preprocess_eye_image(image)
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = diabetes_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    pred_class = predicted.item()

    return {
        "disease": "Diabetic Retinopathy",
        "diagnosis": DIABETES_CLASSES[pred_class],
        "severity": DIABETES_SEVERITY[pred_class],
        "confidence": round(confidence.item() * 100, 2),
        "class_id": pred_class
    }

@app.post("/predict/anemia")
async def predict_anemia(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = anemia_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    pred_class = predicted.item()

    return {
        "disease": "Anemia",
        "diagnosis": ANEMIA_CLASSES[pred_class],
        "severity": "anemic" if pred_class == 0 else "normal",
        "confidence": round(confidence.item() * 100, 2),
        "class_id": pred_class
    }