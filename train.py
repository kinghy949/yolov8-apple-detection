from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs",
        name="apple",
    )


if __name__ == "__main__":
    main()
