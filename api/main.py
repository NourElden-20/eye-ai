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
import tensorflow as tf 

app = FastAPI()

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

# ==================== Diabetes Model (TensorFlow/Keras) ====================
# تحميل الموديل الجديد الذي قمنا بتدريبه
diabetes_model = tf.keras.models.load_model("models/4_best_advanced_model.keras")
print("✅ Advanced Diabetes Model (Keras) loaded!")

# ==================== Anemia Model (PyTorch) ====================
anemia_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
anemia_model.load_state_dict(torch.load("models/best_anemia_model.pth", map_location=DEVICE))
anemia_model = anemia_model.to(DEVICE)
anemia_model.eval()
print("✅ Anemia Model loaded!")

# ==================== Hypertension Model (PyTorch) ====================
def load_hypertension_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("models/hypertension_efficientnet_b0.pth", map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

hypertension_model = load_hypertension_model()
print("✅ Hypertension Model loaded!")

# ==================== PyTorch Transform (لضغط الدم والأنيميا) ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==================== Preprocessing for Diabetes (Ben Graham) ====================
# دالة المعالجة الخاصة بموديل السكري الجديد فقط
def preprocess_for_diabetes(image_bytes):
    # قراءة الصورة من الـ bytes
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # تطبيق فلتر Ben Graham لزيادة الحدة
    img_float = img.astype(np.float32)
    gaussian = cv2.GaussianBlur(img_float, (0, 0), 10.0)
    enhanced = cv2.addWeighted(img_float, 4, gaussian, -4, 128)
    
    # التطبيع (Normalization)
    final_img = enhanced.astype(np.float32) / 255.0
    return np.expand_dims(final_img, axis=0)

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

HYPERTENSION_CLASSES = {
    0: "Hypertensive Retinopathy",
    1: "Normal Fundus"
}

# ==================== Endpoints ====================
@app.get("/")
def root():
    return {"message": "Eye Diagnosis API is running!"}

@app.post("/predict/diabetes")
async def predict_diabetes(file: UploadFile = File(...)):
    contents = await file.read()
    
    # استخدام دالة المعالجة الجديدة الخاصة بـ TensorFlow
    image_tensor = preprocess_for_diabetes(contents)
    
    # التوقع (Prediction)
    predictions = diabetes_model.predict(image_tensor)
    pred_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return {
        "disease"   : "Diabetic Retinopathy",
        "diagnosis" : DIABETES_CLASSES[pred_class],
        "severity"  : DIABETES_SEVERITY[pred_class],
        "confidence": round(confidence * 100, 2),
        "class_id"  : pred_class
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
        "disease"   : "Anemia",
        "diagnosis" : ANEMIA_CLASSES[pred_class],
        "severity"  : "anemic" if pred_class == 0 else "normal",
        "confidence": round(confidence.item() * 100, 2),
        "class_id"  : pred_class
    }

@app.post("/predict/hypertension")
async def predict_hypertension(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = hypertension_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    pred_class = predicted.item()

    return {
        "disease"   : "Hypertensive Retinopathy",
        "diagnosis" : HYPERTENSION_CLASSES[pred_class],
        "severity"  : "hypertensive" if pred_class == 0 else "normal",
        "confidence": round(confidence.item() * 100, 2),
        "class_id"  : pred_class
    }