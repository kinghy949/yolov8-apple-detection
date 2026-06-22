import os
import gradio as gr
from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "runs/apple/weights/best.pt")
if not os.path.exists(WEIGHTS):
    WEIGHTS = "yolov8n.pt"

model = YOLO(WEIGHTS)


def detect(image, conf):
    results = model.predict(source=image, conf=conf, verbose=False)
    plotted = results[0].plot()[:, :, ::-1]  # BGR -> RGB
    count = len(results[0].boxes)
    return plotted, f"检测到 {count} 个苹果"


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="输入图片"),
        gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="置信度阈值"),
    ],
    outputs=[gr.Image(label="识别结果"), gr.Textbox(label="数量")],
    title="YOLOv8 苹果识别",
)

if __name__ == "__main__":
    demo.launch()
