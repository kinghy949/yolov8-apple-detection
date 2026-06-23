import os
import gradio as gr
from ultralytics import YOLO

# 优先用训练产物，未训练时回退到预训练模型（COCO 中已含 apple 类别）
WEIGHTS = os.environ.get("WEIGHTS", "runs/apple/weights/best.pt")
if not os.path.exists(WEIGHTS):
    WEIGHTS = "yolov8n.pt"

model = YOLO(WEIGHTS)


def _summary(boxes):
    """根据检测框生成中文统计文字。"""
    n = len(boxes)
    if n == 0:
        return "未检测到苹果"
    confs = boxes.conf.tolist()
    avg = sum(confs) / n
    return f"检测到 {n} 个苹果（平均置信度 {avg:.2f}，最高 {max(confs):.2f}）"


def detect_image(image, conf):
    if image is None:
        return None, "请先上传图片"
    results = model.predict(source=image, conf=conf, verbose=False)
    r = results[0]
    plotted = r.plot()[:, :, ::-1]  # BGR -> RGB
    return plotted, _summary(r.boxes)


def detect_video(video, conf):
    if video is None:
        return None, "请先上传视频"
    # stream=True 边读边处理，省内存；返回带框视频路径
    results = model.predict(source=video, conf=conf, save=True, verbose=False)
    save_dir = results[0].save_dir
    name = os.path.basename(video)
    out = os.path.join(save_dir, name)
    if not os.path.exists(out):  # 输出可能被转码为 .avi
        base = os.path.splitext(name)[0]
        for f in os.listdir(save_dir):
            if f.startswith(base):
                out = os.path.join(save_dir, f)
                break
    total = sum(len(res.boxes) for res in results)
    return out, f"共 {len(results)} 帧，累计检测框 {total} 个"


with gr.Blocks(title="YOLOv8 苹果识别") as demo:
    gr.Markdown("# 🍎 YOLOv8 苹果识别")

    with gr.Tab("图片"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="numpy", label="输入图片")
                img_conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="置信度阈值")
                img_btn = gr.Button("识别", variant="primary")
            with gr.Column():
                img_out = gr.Image(label="识别结果")
                img_txt = gr.Textbox(label="统计", interactive=False)
        img_btn.click(detect_image, [img_in, img_conf], [img_out, img_txt])

    with gr.Tab("视频"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="输入视频")
                vid_conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="置信度阈值")
                vid_btn = gr.Button("识别", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="识别结果")
                vid_txt = gr.Textbox(label="统计", interactive=False)
        vid_btn.click(detect_video, [vid_in, vid_conf], [vid_out, vid_txt])


if __name__ == "__main__":
    demo.launch()
