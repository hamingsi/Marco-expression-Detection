import sys
import cv2
import numpy as np
import pygame
import os
import logging
from dataset import pil_loader
from utils import *
from conf import get_config, set_logger, set_outdir, set_env
from PIL import Image, ImageDraw, ImageFont

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
    pred = None
    if face_coor is not None:
        # 获取人脸的坐标,并用矩形框出
        [x, y, w, h] = face_coor
        face_image = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        img_ = img_transform(img).unsqueeze(0)

        if torch.cuda.is_available():
            img_ = img_.cuda()

        with torch.no_grad():
            pred = net(img_)
            pred = pred.squeeze().cpu().numpy()

        # 模型预测
        predicted_classes = np.where(pred >= 0.5)[0]
    else:
        predicted_classes = []

    return predicted_classes, pred

def main():
    pygame.init()

    font = pygame.font.SysFont('Arial', 18)
    texts = ['AU1','AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11',
             'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
             'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39',
             'AUL1', 'AUR1', 'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10',
             'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']
    for i in range(len(texts)):
        texts[i] = font.render(texts[i], True, (0, 0, 0))

    screen_width = 1280
    screen_height = 640
    video_width = 640
    video_height = 720/1280*640
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Real-Time Video Classification with Highlight")

    clock = pygame.time.Clock()

    dark_img = pygame.image.load('figs/dark.png').convert_alpha()
    dark_img = pygame.transform.scale(dark_img, (80, 40))
    light_img = pygame.image.load('figs/light.png').convert_alpha()
    light_img = pygame.transform.scale(light_img, (80, 40))

    video_path = 0
    capture = cv2.VideoCapture(video_path)  # 使用摄像头

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = capture.read()
        if ret:
            predicted_classes, pred_prob = classify_and_highlight(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            # 在 PyGame 中显示视频帧
            frame_surface = pygame.surfarray.make_surface(frame)
            frame_surface = pygame.transform.scale(frame_surface, (video_width, video_height))
            screen.blit(frame_surface, (0, 560/1280*640/2))

            for x in range(41):
                i = x // 5
                j = x % 5
                screen.blit(dark_img, (700 + j * 110, 20 + i * 70))

            for x in predicted_classes:
                i = x // 5
                j = x % 5
                screen.blit(light_img, (700 + j * 110, 20 + i * 70))

            for x in range(41):
                i = x // 5
                j = x % 5
                screen.blit(texts[x], (706 + j * 110, 20 + i * 70))

            if pred_prob is not None:
                for x in range(41):
                    i = x // 5
                    j = x % 5
                    tmp_text = font.render('{:.2f}%'.format(pred_prob[x]*100), True, (0, 0, 0))
                    screen.blit(tmp_text, (706 + j * 110, 40 + i * 70))

            pygame.display.flip()
            clock.tick()  # 每秒30帧

    capture.release()
    pygame.quit()

if __name__ == "__main__":

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

    CASC_PATH = './haarcascade_frontalface_alt2.xml'
    cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

    # 定义高亮类别及其对应的矩形框位置
    highlight_classes = {
        0: [(50, 50, 150, 150)],  # 类别0对应的矩形框位置
        1: [(200, 200, 300, 300)],  # 类别1对应的矩形框位置
        # 可以继续添加更多类别和对应的矩形框位置
    }

    main()
