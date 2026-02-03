from ultralytics import YOLO
import torch
import cv2
import os
from pathlib import Path

# ==============================
# Device check
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ==============================
# Load trained model
# ==============================
model = YOLO(
    "C:/Users/Bhaktesh/electronic_component_detection/runs/detect/train18/weights/best.pt"
)

# ==============================
# Batch detect on validation images
# ==============================
def detect_and_save_same_folder(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"❌ Folder not found: {image_dir}")

    image_extensions = {".jpg", ".jpeg", ".png"}

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]

    print(f"📂 Found {len(images)} images in {image_dir}")

    for img_path in images:
        print(f"🔍 Detecting: {img_path.name}")

        results = model.predict(
            source=str(img_path),
            device=device,
            conf=0.6,
            show=False
        )

        annotated = results[0].plot()

        # ✅ Save with SAME name in SAME folder
        save_path = image_dir / img_path.name
        cv2.imwrite(str(save_path), annotated)

    print("✅ All validation images detected and saved in the same folder")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    detect_and_save_same_folder("valid/images")
