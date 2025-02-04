import gradio as gr
import numpy as np
import os
import torch
import torch.nn as nn
import cv2
import logging
from dataset import pil_loader
from utils import *
from conf import get_config, set_logger, set_outdir, set_env
from PIL import Image, ImageDraw, ImageFont

# 假设我们有一个预训练的图像分类模型
conf = get_config()
conf.evaluate = True
set_env(conf)
# generate outdir name
set_outdir(conf)
# Set the logger
set_logger(conf)

dataset_info = hybrid_prediction_infolist

if conf.stage == 1:
    from model.ANFL import MEFARG

    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc,
                 neighbor_num=conf.neighbor_num, metric=conf.metric)
else:
    from model.MEFL import MEFARG

    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)

if conf.resume != '':
    net = load_state_dict(net, conf.resume)

net.eval()
img_transform = image_eval()

if torch.cuda.is_available():
    net = net.cuda()

CASC_PATH = 'C:/Users/ZhouZijian/anaconda3/envs/tf/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

# 定义高亮类别及其对应的矩形框 ID 和显示位置
highlight_classes = {}
for i in range(41):
    highlight_classes[i] = f'box{i}'

# 定义所有41个类别的矩形框
boxes_html = """
<div id="box0" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box1" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box2" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: px;">类别0</div>
<div id="box3" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box4" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box5" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box6" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box7" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box8" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box9" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box10" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box11" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box12" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box13" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box14" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box15" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box16" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box17" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box18" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box19" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box20" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box21" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box22" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box23" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box24" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box25" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box26" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box27" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box28" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box29" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box30" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box31" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box32" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box33" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box34" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box35" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box36" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box37" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box38" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 50px;">类别0</div>
<div id="box39" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
<div id="box40" style="width: 100px; height: 100px; border: 2px solid black; position: absolute; top: 50px; left: 200px;">类别1</div>
"""


def face_detect(image, cascade_classifier):
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图片变为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        # 调整scaleFactor参数的大小，可以增加识别的灵敏度，推荐1.1
        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    # 如果图片中没有检测到人脸，则返回None
    if not len(faces) > 0:
        return None
    # max_are_face包含了人脸的坐标，大小
    max_are_face = faces[0]
    # 在所有人脸中选一张最大的脸
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    # 这两步可有可无
    face_coor = max_are_face
    return face_coor


def classify_and_highlight(frame):
    # 将帧转换为模型输入格式
    face_coor = face_detect(frame, cascade_classifier)
    print(face_coor)
    if face_coor is not None:
        # 获取人脸的坐标,并用矩形框出
        [x, y, w, h] = face_coor
        face_image = frame[y:y + h, x:x + w]
        img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        img_ = img_transform(img).unsqueeze(0)

        if torch.cuda.is_available():
            img_ = img_.cuda()

        with torch.no_grad():
            pred = net(img_)
            pred = pred.squeeze().cpu().numpy()

        # 模型预测
        predicted_classes = np.where(pred >= 0.5)[0]

        # 生成需要高亮的矩形框 ID 列表
        highlight_box_ids = [highlight_classes.get(cls, '') for cls in predicted_classes]

        return highlight_box_ids
    else:
        return []


def update_html(box_ids):
    updated_html = boxes_html
    for box_id in box_ids:
        if box_id:
            updated_html += f'<script>document.getElementById("{box_id}").style.border="2px solid red";</script>'
    return updated_html


def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box_ids = classify_and_highlight(frame)
        yield update_html(box_ids)
    cap.release()


# 创建 Gradio 接口
video_input = gr.Video()
html_output = gr.HTML()

demo = gr.Interface(
    fn=process_video,
    inputs=video_input,
    outputs=html_output,
    live=True
)

demo.launch()
