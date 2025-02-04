import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import logging
from dataset import pil_loader
from utils import *
from conf import get_config, set_logger, set_outdir, set_env
from PIL import Image, ImageDraw, ImageFont



def main(conf):
    dataset_info = hybrid_prediction_infolist

    # data
    img_path = conf.input

    if conf.stage == 1:
        from model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc,
                     neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    net.eval()
    img_transform = image_eval()
    img = pil_loader(img_path)
    img_ = img_transform(img).unsqueeze(0)

    # if torch.cuda.is_available():
    #     net = net.cuda()
    #     img_ = img_.cuda()

    with torch.no_grad():
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()

    # log
    infostr = {'AU prediction:'}
    logging.info(infostr)
    infostr_probs, infostr_aus = dataset_info(pred, 0.5)
    logging.info(infostr_aus)
    logging.info(infostr_probs)

    if conf.draw_text:
        img = draw_text(conf.input, list(infostr_aus), pred)
        import cv2
        path = conf.input.split('.')[0] + '_pred.jpg'
        cv2.imwrite(path, img)


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

def put_Chinese(im, text, x, y, color, font):
    cv2_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
    draw.text((x, y), text, color, font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return cv2_text_im


def main_video(videoPath):
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") # 第一个参数为字体文件路径，第二个为字体大小
    dataset_info = hybrid_prediction_infolist

    # data
    img_path = conf.input

    if conf.stage == 1:
        from model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc,
                     neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    net.eval()
    img_transform = image_eval()

    # if torch.cuda.is_available():
    #     net = net.cuda()

    CASC_PATH = 'C:/Users/ZhouZijian/anaconda3/envs/tf/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
    cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
    video_captor = cv2.VideoCapture(videoPath)
    fps = video_captor.get(cv2.CAP_PROP_FPS)
    size = (int(video_captor.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_captor.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    totalFrames = int(video_captor.get(7))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # 创建新视频
    ViideoWrite = cv2.VideoWriter("test2.mp4", fourcc, fps, size)
    while True:
        # 获取摄像头的每帧图片，若获得，则ret的值为True,frame就是每一帧的图像，是个三维矩阵
        ret, frame = video_captor.read()
        face_coor = face_detect(frame, cascade_classifier)

        if face_coor is not None:
            # 获取人脸的坐标,并用矩形框出
            [x, y, w, h] = face_coor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if cv2.waitKey(1):
            if face_coor is not None:
                [x, y, w, h] = face_coor
                face_image = frame[y:y + h, x:x + w]
                img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

                img_ = img_transform(img).unsqueeze(0)

                # if torch.cuda.is_available():
                #     img_ = img_.cuda()

                with torch.no_grad():
                    pred = net(img_)
                    pred = pred.squeeze().cpu().numpy()

                infostr_probs, infostr_aus = dataset_info(pred, 0.5)
                infostr_probs = list(infostr_probs)[0].split(' ')
                infostr_probs = list(filter(None, infostr_probs))
                bias = 80
                for index, au in enumerate(infostr_aus):
                    put_Chinese(frame, au, 10, bias + index * 20 + 20, (0, 0, 255), font)
                    # cv2.putText(frame, au, (10, bias + index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                for index in range(0, len(infostr_probs)-1, 2):

                    cv2.putText(frame, infostr_probs[index] + infostr_probs[index+1], (450, bias + index * 10 + 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

                # # log
                # infostr = {'AU prediction:'}
                # logging.info(infostr)
                # infostr_probs, infostr_aus = dataset_info(pred, 0.5)
                # logging.info(infostr_aus)
                # logging.info(infostr_probs)

        cv2.imshow('face', frame)
        ViideoWrite.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
    # main_video('test_video/test.mp4')
