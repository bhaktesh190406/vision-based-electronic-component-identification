from ultralytics import YOLO

def train_model():
    model = YOLO("yolov11n.pt")  # or yolov11s.pt
    model.train(
        data="dataset/data.yaml",
        epochs=75,
        imgsz=640,
        batch=20,
        device=0  # 👈 FORCE GPU
    )

if __name__ == "__main__":
    train_model()
