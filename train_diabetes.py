import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# ==================== Config ====================
DATA_DIR = "data/aptos2019/train_images"
CSV_PATH = "data/aptos2019/train.csv"
MODEL_SAVE = "models/diabetes_model.pth"
EPOCHS = 15
BATCH_SIZE = 16
LR = 0.0001
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using: {DEVICE}")

# ==================== Dataset ====================
class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['id_code'] + '.png')
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['diagnosis']
        if self.transform:
            image = self.transform(image)
        return image, label

# ==================== Transforms ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==================== Load Data ====================
df = pd.read_csv(CSV_PATH)

train_size = int(0.8 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]

train_loader = DataLoader(RetinaDataset(train_df, DATA_DIR, transform),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RetinaDataset(val_df, DATA_DIR, transform),
                        batch_size=BATCH_SIZE)

# ==================== Model ====================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==================== Training ====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_df) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Acc: {acc:.2f}%")

# ==================== Save ====================
torch.save(model.state_dict(), MODEL_SAVE)
print(f"Model saved to {MODEL_SAVE}")