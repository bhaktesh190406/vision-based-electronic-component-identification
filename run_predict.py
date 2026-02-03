from ultralytics import YOLO
import os

print("Script started")
print("Current folder:", os.getcwd())

model = YOLO("runs/detect/train18/weights/best.pt")

results = model.predict(
    source="valid/images",
    conf=0.25,
    save=True
)

print("DONE! Check runs/detect/predict/")
print("Predictions saved to:", results[0].path)