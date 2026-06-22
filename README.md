# yolov8-apple-detection

基于 YOLOv8 的苹果识别系统，简单可用。

## 环境

```bash
pip install -r requirements.txt
```

## 数据准备

按 YOLO 格式放入 `datasets/apple/`：

```
datasets/apple/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt   # 每行: 0 cx cy w h（归一化坐标）
  labels/val/*.txt
```

类别只有一个：`apple`（见 `data.yaml`）。

## 训练

```bash
python train.py
```

权重输出在 `runs/apple/weights/best.pt`。

## 推理

```bash
# 图片
python detect.py path/to/img.jpg --save

# 摄像头
python detect.py 0
```

未训练前可直接用预训练模型试跑：

```bash
python detect.py path/to/img.jpg --weights yolov8n.pt --save
```

## Web 演示

```bash
python app.py
```

打开浏览器上传图片即可识别。
