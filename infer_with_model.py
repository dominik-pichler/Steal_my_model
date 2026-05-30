import sys
import numpy as np
import torch
import torchvision.models as M
import cv2

# 1) Load the reconstructed model
model = M.resnet18(num_classes=1)
model.load_state_dict(torch.load("reconstructed_state_dict.pth"))
model.eval()

# 2) Read image with the same BGR convention the binary uses
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
if img is None:
    print(f"Failed to load file: {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
h, w = img.shape[:2]

# 3) Resize: shortest side to 256 (matching the binary's manual implementation)
factor = max(256 / h, 256 / w)
new_w = round(w * factor)
new_h = round(h * factor)
img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# 4) Center crop to 224×224
y0 = (new_h - 224) // 2
x0 = (new_w - 224) // 2
img = img[y0:y0+224, x0:x0+224]

# 5) BGR (uint8) → planar RGB FP32, normalized with ImageNet mean/std
img = img.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img = (img[..., ::-1] - mean) / std            # BGR→RGB and normalize
img = img.transpose(2, 0, 1)                    # HWC → CHW
tensor = torch.from_numpy(img.copy()).unsqueeze(0)  # add batch dim → (1, 3, 224, 224)

# 6) Forward pass
with torch.no_grad():
    logit = model(tensor).item()

# 7) Same decision rule as the binary
prediction = int(logit < 0)
print(prediction)