import cv2
import pygame
from utils import *
from PIL import Image
import multiprocessing
import mediapipe as mp
from au2emotion import au_to_facial_region
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import onnxruntime

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class image_eval_with_numpy:
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, img):
        w, h = img.size
        if w > h:
            w = round(w/h * self.img_size)
            h = self.img_size
        else:
            h = round(h/w * self.img_size)
            w = self.img_size
        # Resize
        img = img.resize((w, h), Image.BILINEAR)

        # Center crop
        left = round((w - self.crop_size) / 2)
        top = round((h - self.crop_size) / 2)
        right = left + self.crop_size
        bottom = top + self.crop_size
        img = img.crop((left, top, right, bottom))

        # Convert to numpy array and normalize
        img = np.array(img).astype(np.float32) / 255
        img = (img - self.mean) / self.std

        # Transpose to match PyTorch's CHW format
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        return img.astype(np.float32)



def my_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def draw_histogram(screen, hist, hist_x, hist_y, hist_width, hist_height, num_bins, c_font):
    # 绘制直方图
    bar_width = hist_width / num_bins
    if hist is not None:
        for i in range(num_bins):
            bar_height = hist[i] * (hist_height - 60)  # 留出空间给标签
            x = hist_x + i * bar_width
            y = hist_y + hist_height - bar_height - 30  # 30是底部标签的空间
            pygame.draw.rect(screen, RED, (x, y, bar_width, bar_height))

    labels = ["愤怒", "蔑视", "厌恶", "恐惧", "高兴", "中立", "悲伤", '惊讶']

    for i in range(num_bins + 1):
        x = hist_x + i * bar_width
        pygame.draw.line(screen, BLACK, (x, hist_y + hist_height - 30), (x, hist_y + hist_height - 20))

        if i < num_bins:
            label = labels[i]
            text = c_font.render(label, True, BLACK)
            text_rect = text.get_rect()
            # 将标签放置在每个bin的中心
            text_rect.centerx = x + bar_width / 2
            text_rect.bottom = hist_y + hist_height - 5
            screen.blit(text, text_rect)


def face_detect(lock, frame_queue, process_queue):
    mp_face_detection = mp.solutions.face_detection
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    while True:
        # print('face_detect')
        if not frame_queue.empty():
            img_ori = frame_queue.get()
            img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]

            with mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5) as face_detection:

                results = face_detection.process(img_ori)
                if results.detections is not None:
                    detection_box = results.detections[0].location_data.relative_bounding_box
                    xmin, width, ymin, height = detection_box.xmin, detection_box.width, detection_box.ymin, detection_box.height
                    face_coor = [int(xmin * W), int(ymin * H - 0.2 * height * H), int(width * W), int(height * 1.2 * H)]
                    landmarks = {}
                    a = lambda x, i: x.detections[0].location_data.relative_keypoints[i]
                    # eye
                    l_ex = a(results, 0).x
                    l_ey = a(results, 0).y
                    r_ex = a(results, 1).x
                    r_ey = a(results, 1).y
                    landmarks['eye'] = [[l_ex * W, l_ey * H], [r_ex * W, r_ey * H]]
                    # nose
                    n_x = a(results, 2).x
                    n_y = a(results, 2).y
                    landmarks['nose'] = [[n_x * W, n_y * H]]
                    # mouth
                    m_x = a(results, 3).x
                    m_y = a(results, 3).y
                    landmarks['lip'] = [[m_x * W, m_y * H]]
                    # cheek
                    l_cx = a(results, 4).x
                    l_cy = a(results, 4).y
                    r_cx = a(results, 5).x
                    r_cy = a(results, 5).y
                    landmarks['cheek'] = [[l_cx * W, l_cy * H], [r_cx * W, r_cy * H]]
                    # identity
                    l2_norm = lambda x, y: (x ** 2 + y ** 2) ** (1 / 2)
                    e_x = n_x - m_x
                    e_y = n_y - m_y
                    e_norm = l2_norm(e_x, e_y)
                    e_x /= e_norm
                    e_y /= e_norm
                    # brow
                    l_bx = l_ex + l2_norm(l_ex - n_x, l_ey - n_y) * 0.7 * e_x
                    l_by = l_ey + l2_norm(l_ex - n_x, l_ey - n_y) * 0.7 * e_y
                    r_bx = r_ex + l2_norm(r_ex - n_x, r_ey - n_y) * 0.7 * e_x
                    r_by = r_ey + l2_norm(r_ex - n_x, r_ey - n_y) * 0.7 * e_y
                    landmarks['brow'] = [[l_bx * W, l_by * H], [r_bx * W, r_by * H]]
                    # Lip corner
                    llc_x = l_ex - e_norm * 2 * e_x
                    llc_y = l_ey - e_norm * 2 * e_y
                    rlc_x = r_ex - e_norm * 2 * e_x
                    rlc_y = r_ey - e_norm * 2 * e_y
                    landmarks['lip_corner'] = [[llc_x * W, llc_y * H], [rlc_x * W, rlc_y * H]]
                    # chin
                    ch_x = m_x - e_x * e_norm
                    ch_y = m_y - e_y * e_norm
                    landmarks['chin'] = [[ch_x * W, ch_y * H]]

                    with lock:
                        if process_queue.full():
                            process_queue.get()
                        process_queue.put((face_coor, img_ori, landmarks))

                else:
                    with lock:
                        if process_queue.full():
                            process_queue.get()
                        process_queue.put((None, img_ori, None))


def classify_and_highlight(lock, process_queue, result_queue):
    # 加载ONNX模型
    ort_session = onnxruntime.InferenceSession("graph_au_model.onnx", providers=['CoreMLExecutionProvider'])
    img_transform = image_eval_with_numpy()

    while True:
        # print('classify_and_highlight')
        if not process_queue.empty():
            face_coor, frame, landmarks = process_queue.get()
            if face_coor is not None:
                [x, y, w, h] = face_coor
                face_image = frame[max(y, 0):y + h, max(x, 0):x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                img = img_transform(img)

                input_name = ort_session.get_inputs()[0].name
                ort_inputs = {input_name: np.ascontiguousarray(img)}
                pred = ort_session.run(None, ort_inputs)[0][0]

                # 模型预测
                predicted_classes = np.where(pred >= 0.5)[0]
                regions = au_to_facial_region(predicted_classes)
                for r in regions:
                    for point in landmarks[r]:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), 2)
                with lock:
                    if result_queue.full():
                        result_queue.get()
                    result_queue.put((frame, predicted_classes, pred, face_image))
            else:
                with lock:
                    if result_queue.full():
                        result_queue.get()
                    result_queue.put((frame, None, None, None))


def emotion_detector(lock, result_queue, result_queue2):
    model_name = 'enet_b0_8_best_afew'
    fer = HSEmotionRecognizer(model_name=model_name)
    while True:
        # print('emotion_detector')
        if not result_queue.empty():
            frame, predicted_classes, pred, face_image = result_queue.get()
            if predicted_classes is not None:
                fe_res, fe_hist = fer.predict_emotions(face_image, logits=True)
                fe_hist = my_softmax(fe_hist)

                with lock:
                    if result_queue2.full():
                        result_queue2.get()
                    result_queue2.put((frame, predicted_classes, pred, fe_res, fe_hist))
            else:
                with lock:
                    if result_queue2.full():
                        result_queue2.get()
                    result_queue2.put((frame, predicted_classes, pred, None, None))


def draw(lock, result_queue2):
    pygame.init()

    font = pygame.font.SysFont('Arial', 18)
    c_font = pygame.font.SysFont('pingfang', 18)
    # texts = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11',
    #          'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
    #          'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39',
    #          'AUL1', 'AUR1', 'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10',
    #          'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']
    texts = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11',
             'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
             'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39']
    aus = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11',
           'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
           'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39']
    au_num = len(texts)

    au_dict = {
        "AU1": "内眉提升",
        "AU2": "外眉提升",
        "AU4": "眉头下降",
        "AU5": "上眼睑提升",
        "AU6": "颧肌提升",
        "AU7": "眼睑收紧",
        "AU9": "鼻子皱起",
        "AU10": "上唇提升",
        "AU11": "鼻唇沟加深",
        "AU12": "嘴角拉起",
        "AU13": "尖锐唇部拉起",
        "AU14": "嘴角凹陷",
        "AU15": "嘴角下垂",
        "AU16": "下唇下压",
        "AU17": "下巴提升",
        "AU18": "嘴唇撅起",
        "AU19": "吐舌",
        "AU20": "嘴唇拉伸",
        "AU22": "嘴唇漏斗状",
        "AU23": "嘴唇收紧",
        "AU24": "嘴唇压紧",
        "AU25": "嘴唇分开",
        "AU26": "下巴下降",
        "AU27": "嘴巴拉伸",
        "AU32": "咬唇",
        "AU38": "鼻孔扩张",
        "AU39": "鼻孔压缩"
    }

    emotion_dict = {
        "Anger": "愤怒",
        "Contempt": "蔑视",
        "Disgust": "厌恶",
        "Fear": "恐惧",
        "Happiness": "高兴",
        "Neutral": "中立",
        "Sadness": "悲伤",
        'Surprise': '惊讶',
    }

    for i in range(len(texts)):
        texts[i] = font.render(texts[i], True, (0, 0, 0))

    screen_width = 1280
    screen_height = 640

    # 左侧区域设置
    left_area_width = 640
    left_area_height = 640

    # 视频区域设置（保持16:9比例）
    video_width = left_area_width
    video_height = int(video_width * 9 / 16)
    video_x = 0
    video_y = (int(left_area_height * 2 / 3) - video_height) // 2  # 居中放置在上2/3区域

    # 柱状图区域设置（占左侧区域的下方1/3）
    hist_height = left_area_height - int(left_area_height * 2 / 3)
    hist_width = left_area_width
    hist_x = 0
    hist_y = int(left_area_height * 2 / 3)

    num_bins = 8

    # 右侧AU区域设置
    au_area_x = left_area_width + 60
    au_area_y = 20

    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Real-Time Video Classification with Highlight")

    clock = pygame.time.Clock()

    dark_img = pygame.image.load('figs/dark.png').convert_alpha()
    dark_img = pygame.transform.scale(dark_img, (80, 40))
    light_img = pygame.image.load('figs/light.png').convert_alpha()
    light_img = pygame.transform.scale(light_img, (80, 40))
    icon_img = pygame.image.load('figs/icon.png').convert_alpha()
    icon_img = pygame.transform.scale(icon_img, (520, 65))

    running = True
    while running:
        # print('draw')
        if not result_queue2.empty():
            screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            mouse_x, mouse_y = pygame.mouse.get_pos()

            au_idx = -1
            mouse_x -= au_area_x
            mouse_y -= au_area_y
            if mouse_x >= 0 and mouse_y >= 0:
                mouse_x_bias = mouse_x % 110
                mouse_y_bias = mouse_y % 70
                if mouse_x_bias <= 80 and mouse_y_bias <= 40:
                    mouse_row = mouse_x // 110
                    mouse_col = mouse_y // 70
                    au_idx = mouse_col * 5 + mouse_row

            if 0 <= au_idx < au_num:
                locked_au = aus[au_idx]
                au_info = au_dict[locked_au]
                tmp_text = c_font.render(f'动作单元：{au_info}', True, (0, 0, 0))
                text_rect = tmp_text.get_rect()
                # 将标签放置在每个bin的中心
                text_rect.centerx = 960
                text_rect.bottom = 480
                screen.blit(tmp_text, text_rect)

            frame, predicted_classes, pred_prob, fe_res, fe_hist = result_queue2.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            # 在 PyGame 中显示视频帧
            frame_surface = pygame.surfarray.make_surface(frame)
            frame_surface = pygame.transform.scale(frame_surface, (video_width, video_height))
            screen.blit(frame_surface, (video_x, video_y))

            # 绘制AU区域
            for x in range(au_num):
                i = x // 5
                j = x % 5
                screen.blit(dark_img, (au_area_x + j * 110, au_area_y + i * 70))

            if predicted_classes is not None:
                for x in predicted_classes:
                    i = x // 5
                    j = x % 5
                    screen.blit(light_img, (au_area_x + j * 110, au_area_y + i * 70))

            for x in range(au_num):
                i = x // 5
                j = x % 5
                screen.blit(texts[x], (au_area_x + 6 + j * 110, au_area_y + i * 70))

            if pred_prob is not None:
                for x in range(au_num):
                    i = x // 5
                    j = x % 5
                    tmp_text = font.render('{:.2f}%'.format(pred_prob[x] * 100), True, (0, 0, 0))
                    screen.blit(tmp_text, (au_area_x + 6 + j * 110, au_area_y + 20 + i * 70))

            if fe_hist is not None:
                # 绘制情绪文本
                tmp_emotion = c_font.render('情感： ' + emotion_dict[fe_res], True, (0, 0, 0))
            else:
                tmp_emotion = c_font.render('情感：', True, (0, 0, 0))
            screen.blit(tmp_emotion, (hist_x + 10, hist_y + 10))

            # 绘制柱状图
            draw_histogram(screen, fe_hist, hist_x, hist_y, hist_width, hist_height, num_bins, c_font)

            # draw icon
            icon_rect = icon_img.get_rect()
            icon_rect.centerx = 960
            icon_rect.bottom = 620
            screen.blit(icon_img, icon_rect)

            pygame.display.flip()
            print(clock.get_fps())
            clock.tick(30)  # 每秒60帧
    pygame.quit()


def input_read(lock, frame_queue):
    # 使用摄像头
    video_path = 0
    capture = cv2.VideoCapture(video_path)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 30)
    print(capture.get(cv2.CAP_PROP_FPS))
    while True:
        # print('input_read')
        ret, frame = capture.read()
        if ret:
            with lock:
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(frame)
    capture.release()


def main():
    frame_queue = multiprocessing.Queue(maxsize=5)
    process_queue = multiprocessing.Queue(maxsize=5)
    result_queue = multiprocessing.Queue(maxsize=5)
    result_queue2 = multiprocessing.Queue(maxsize=5)
    lock = multiprocessing.Lock()

    input_process = multiprocessing.Process(target=input_read, args=(lock, frame_queue))
    frame_process = multiprocessing.Process(target=face_detect, args=(lock, frame_queue, process_queue))
    process_process = multiprocessing.Process(target=classify_and_highlight, args=(lock, process_queue, result_queue))
    emotion_process = multiprocessing.Process(target=emotion_detector, args=(lock, result_queue, result_queue2))
    result_process = multiprocessing.Process(target=draw, args=(lock, result_queue2))

    input_process.start()
    frame_process.start()
    process_process.start()
    emotion_process.start()
    result_process.start()

    input_process.join()
    frame_process.join()
    process_process.join()
    emotion_process.join()
    result_process.join()


if __name__ == "__main__":
    main()
