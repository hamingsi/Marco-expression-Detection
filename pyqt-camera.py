import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import multiprocessing
import torch
from blazeface import BlazeFace
from dataset import pil_loader
from utils import *
from conf import get_config, set_logger, set_outdir, set_env
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

# 配置
conf = get_config()
conf.evaluate = True
set_env(conf)
set_outdir(conf)
set_logger(conf)

if conf.stage == 1:
    from model.ANFL import MEFARG
    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
else:
    from model.MEFL import MEFARG
    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)

if conf.resume != '':
    net = load_state_dict(net, conf.resume)

net.eval()
img_transform = image_eval()

gpu = torch.device("cpu")
net.to(torch.device('mps'))
front_net = BlazeFace().to(gpu)
front_net.load_weights("./checkpoints/blazeface.pth")
front_net.load_anchors("./checkpoints/anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("./checkpoints/blazefaceback.pth")
back_net.load_anchors("./checkpoints/anchorsback.npy")
front_net.min_score_thresh = 0.75
front_net.min_suppression_threshold = 0.3

class VideoCaptureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.au_list = ['AU1','AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11',
             'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
             'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39',
             'AUL1', 'AUR1', 'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10',
             'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']  # 示例AU列表
        self.pred = None
        
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # 调用检测和分类方法
            frame = self.detect_and_classify(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            q_img = QImage(frame.data, width, height, step, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.update()
            
    def detect_and_classify(self, frame):
        # 在这里添加你的检测和分类代码
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        W = img.shape[1]
        H = img.shape[0]
        img = cv2.resize(img, (128, 128))
        
        front_detections = front_net.predict_on_image(img)
        if front_detections.size(0) != 0:
            [ymin,xmin,ymax,xmax] = front_detections[0,:4]
            center_y = (ymin + ymax)/2
            center_x = (xmin + xmax)/2
            l = (xmax-xmin)/2
            
            face_coor = [int(W*(center_x-l)), int(H*center_y-W*l), int(W*l*2),int(W*l*2)]
        [x, y, w, h] = face_coor
        face_image = frame[max(y,0):y + h, max(x,0):x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        img_ = img_transform(img).unsqueeze(0)
        img_ = img_.to(torch.device("mps"))

        with torch.no_grad():
            pred = net(img_)
            self.pred = pred.squeeze().cpu().numpy()

        # 模型预测
        self.predicted_classes = np.where(self.pred >= 0.5)[0]
        return frame
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 在这里绘制AU和文本
        y_offset = 50
        painter.setFont(QFont('Arial', 16))
        if self.pred is not None:
            for i, au in enumerate(self.au_list):
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(700, y_offset + i * 30, f"{au}: {self.pred[i]*100:.2f}%")
    
    def closeEvent(self, event):
        self.capture.release()
        super().closeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 OpenCV Video Capture")
        self.setGeometry(100, 100, 800, 600)
        
        self.video_capture_widget = VideoCaptureWidget(self)
        self.setCentralWidget(self.video_capture_widget)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
