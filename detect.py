import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="苹果识别推理")
    parser.add_argument("source", help="图片/视频/目录路径，或摄像头编号（如 0）")
    parser.add_argument("--weights", default="runs/apple/weights/best.pt",
                        help="模型权重，默认用训练产物，未训练时可填 yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", help="保存可视化结果")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(source=args.source, conf=args.conf, save=args.save, show=False)


if __name__ == "__main__":
    main()
