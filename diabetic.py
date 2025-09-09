# Diabetic_Retinopathy_PyTorch.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============== CONFIGURATION AND PATHS ===============
# ---- Paths ----
DATA_DIR = "./data/train_images"
LABELS_CSV = "./data/train.csv"
OUT_DIR = "./DR_Phase3_CustomCNN" # New output folder for this version
os.makedirs(OUT_DIR, exist_ok=True)
SYN_DIR = os.path.join(OUT_DIR, "synthetic_cgan")
os.makedirs(SYN_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Hyperparameters ----
IMG_SIZE = 224
LATENT_DIM = 100
COND_DIM = 2
BATCH_SIZE = 64
EPOCHS_CGAN = 100
EPOCHS_CLS = 30

# ---- GPU Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============== 1. DATA LOADING ===============
df = pd.read_csv(LABELS_CSV)
df['DR_PRESENT'] = (df['diagnosis'] > 0).astype(int)

dr_ids = df[df['DR_PRESENT'] == 1]['id_code'].values
nodr_ids = df[df['DR_PRESENT'] == 0]['id_code'].values

classifier_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class RetinopathyDataset(Dataset):
    def __init__(self, ids, directory, transform=None, limit=None):
        self.ids = ids[:limit] if limit else ids
        self.directory = directory
        self.transform = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_name = self.ids[idx]
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name = f"{img_name}.png"
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

generator_path = os.path.join(OUT_DIR, "cgan_generator.pth")
if not os.path.exists(generator_path) or not os.path.exists(os.path.join(OUT_DIR, "synthetic_index.csv")):
    exit(f"Error: GAN model or synthetic data not found in {OUT_DIR}. Please copy them from a previous run.")

print("✅ Found existing GAN data. Proceeding.")


# =============== 5. BUILD FINAL DATASET FOR CLASSIFIER ===============
print(f"Loading all data with {IMG_SIZE}x{IMG_SIZE} resolution for classifier...")

dr_dataset = RetinopathyDataset(dr_ids, DATA_DIR, transform=classifier_transform)
nodr_dataset = RetinopathyDataset(nodr_ids, DATA_DIR, transform=classifier_transform)
X_real_dr = torch.stack([s for s in tqdm(dr_dataset, desc="Loading real DR images")])
X_real_nodr = torch.stack([s for s in tqdm(nodr_dataset, desc="Loading real No-DR images")])

syn_df = pd.read_csv(os.path.join(OUT_DIR, "synthetic_index.csv"))
syn_dataset = RetinopathyDataset(syn_df['filename'].values, SYN_DIR, transform=classifier_transform)
X_syn_dr = torch.stack([s for s in tqdm(syn_dataset, desc="Loading synthetic DR images")])

def sample_hba1c(n): return 6.0 + 8.0 * np.random.beta(5, 2, size=n)
def sample_duration(n): return 2.0 + 28.0 * np.random.beta(4, 3, size=n)
dr_risk = np.stack([sample_hba1c(len(X_real_dr))/15.0, sample_duration(len(X_real_dr))/30.0], axis=1).astype(np.float32)
nodr_risk = np.zeros((len(X_real_nodr), 2), dtype=np.float32)
syn_risk_np = syn_df[["HbA1c", "Duration"]].values
syn_risk = np.stack([syn_risk_np[:,0]/15.0, syn_risk_np[:,1]/30.0], axis=1).astype(np.float32)

X_imgs = torch.cat([X_real_dr, X_syn_dr, X_real_nodr], axis=0)
R_risk = np.concatenate([dr_risk, syn_risk, nodr_risk], axis=0)
y_dr = np.concatenate([np.ones(len(X_real_dr)), np.ones(len(X_syn_dr)), np.zeros(len(X_real_nodr))], axis=0)

Xtr_idx, Xval_idx, Rtr, Rval, ydr_tr, ydr_val = train_test_split(
    range(len(X_imgs)), R_risk, y_dr, test_size=0.2, stratify=y_dr, random_state=42
)
Xtr, Xval = X_imgs[Xtr_idx], X_imgs[Xval_idx]

class FinalClassifierDataset(Dataset):
    def __init__(self, images_tensor, risks_np, dr_labels_np):
        self.images = images_tensor
        self.risks = torch.from_numpy(risks_np).float()
        self.dr_labels = torch.from_numpy(dr_labels_np).float().unsqueeze(1)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.risks[idx], self.dr_labels[idx]

train_dataset = FinalClassifierDataset(Xtr, Rtr, ydr_tr)
val_dataset = FinalClassifierDataset(Xval, Rval, ydr_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============== 6. MODEL: CUSTOM VGG-STYLE CNN ===============
count_nodr = (ydr_tr == 0).sum(); count_dr = (ydr_tr == 1).sum()
weight_for_dr = count_nodr / count_dr
pos_weight = torch.tensor([weight_for_dr], device=device)
print(f"Calculated pos_weight for DR class: {pos_weight.item():.2f}")

class VGGStyleCNN(nn.Module):
    def __init__(self):
        super(VGGStyleCNN, self).__init__()
        self.block1 = self._make_block(3, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.block4 = self._make_block(128, 256)
        self.final_conv_block = self._make_block(256, 512, pool=False)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.risk_processor = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    def _make_block(self, in_channels, out_channels, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if pool: layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    def forward(self, img, risk):
        x = self.final_conv_block(self.block4(self.block3(self.block2(self.block1(img)))))
        img_features = torch.flatten(self.avgpool(x), 1)
        risk_features = self.risk_processor(risk)
        return self.head(torch.cat([img_features, risk_features], dim=1))

model = VGGStyleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("Training final classifier (Custom VGG-Style CNN)...")
for epoch in range(1, EPOCHS_CLS + 1):
    model.train()
    total_loss = 0
    for img_b, risk_b, dr_l in train_loader:
        img_b, risk_b, dr_l = img_b.to(device), risk_b.to(device), dr_l.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(img_b, risk_b), dr_l)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS_CLS} | Train Loss: {total_loss / len(train_loader):.4f}")

print("✅ Classifier training complete.")

# =============== 7. GRAD-CAM IMPLEMENTATION AND VISUALIZATION ===============
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.feature_maps = None; self.gradients = None
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_backward_hook(self.save_gradients)
    def save_feature_maps(self, module, input, output): self.feature_maps = output.detach()
    def save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0].detach()
    def __call__(self, img_tensor, risk_tensor):
        self.model.eval()
        score = self.model(img_tensor, risk_tensor)
        self.model.zero_grad()
        score.backward()
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.feature_maps.shape[1]): self.feature_maps[:, i, :, :] *= pooled_grads[i]
        heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1e-8
        return heatmap.numpy()

grad_cam = GradCAM(model, target_layer=model.final_conv_block)

print("Visualizing Grad-CAM on a sample DR image...")
dr_val_idx = np.where(ydr_val == 1)[0][0]
sample_img_tensor = Xval[dr_val_idx].unsqueeze(0).to(device)
sample_risk_tensor = torch.from_numpy(Rval[dr_val_idx]).unsqueeze(0).to(device)
sample_img_np = Xval[dr_val_idx].cpu().numpy().transpose(1, 2, 0)
heatmap = grad_cam(sample_img_tensor, sample_risk_tensor)

def overlay_heatmap(img, heatmap, alpha=0.4, cmap=cv2.COLORMAP_JET):
    img_uint8 = (img * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)
    overlay = cv2.addWeighted(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

overlay = overlay_heatmap(sample_img_np, heatmap)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(sample_img_np); axs[0].set_title("Original DR Image"); axs[0].axis("off")
axs[1].imshow(overlay); axs[1].set_title("Grad-CAM Overlay"); axs[1].axis("off")
plt.tight_layout()
grad_cam_save_path = os.path.join(RESULTS_DIR, "grad_cam_visualization_custom.png")
plt.savefig(grad_cam_save_path)
print(f"Saved Grad-CAM visualization to {grad_cam_save_path}")
plt.close(fig)

# =============== 8. FINAL MODEL EVALUATION ===============
model.eval()
all_preds = []
with torch.no_grad():
    for img_b, risk_b, _ in val_loader:
        img_b, risk_b = img_b.to(device), risk_b.to(device)
        pred_dr = model(img_b, risk_b)
        all_preds.extend(torch.sigmoid(pred_dr).cpu().numpy())

y_pred = (np.array(all_preds) > 0.5).astype("int32").flatten()
y_true = ydr_val

print("--- Final Model Evaluation ---")
report = classification_report(y_true, y_pred, target_names=['No DR', 'DR Present'])
print("Classification Report:")
print(report)

report_save_path = os.path.join(RESULTS_DIR, "classification_report_custom.txt")
with open(report_save_path, 'w') as f: f.write(report)
print(f"Saved classification report to {report_save_path}")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No DR', 'DR Present'], yticklabels=['No DR', 'DR Present'], ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Final Model Confusion Matrix (Custom CNN)')
cm_save_path = os.path.join(RESULTS_DIR, "confusion_matrix_custom.png")
plt.savefig(cm_save_path)
print(f"Saved confusion matrix to {cm_save_path}")
plt.close(fig)

print(f"\nAll requested results have been saved to the new '{OUT_DIR}/results/' folder.")