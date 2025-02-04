import numpy as np
import numpy as np

def au_to_facial_region(au_numbers):
    au_numbers = au_numbers + 1
    # 定义AU到面部区域的映射关系
    au_mapping = {
        'brow': [1, 2, 4],    # 眉毛区域
        'eye': [5, 6, 7],     # 眼睛区域
        'nose': [9,38,39],          # 鼻子区域
        'cheek': [6],         # 脸颊区域（与眼睛区域共享AU6）
        'lip_corner': [12, 15],  # 嘴角区域
        'chin': [17, 26],     # 下巴区域
        'lip': [10,16,18, 20, 22, 23, 24, 25, 27],  # 嘴唇\嘴巴区域（与嘴唇区域共享AU27）
    }
    # 将AU编号转换为面部区域
    regions = []
    for au in au_numbers:
        for region, aus in au_mapping.items():
            if au in aus:
                regions.append(region)
                break
    return np.array(regions)

def classify_expression(aus):
    # 定义情绪与AU的映射关系，包括独有的AU集合和共享的AU集合
        # emotion_aus = {
    #     'Happiness': [{6,12,28}, {1, 14, 26, 27}],
    #     'Surprise': [{2},{1, 5, 25, 26, 27}],
    #     'Anger': [{16,22,23},{4, 9, 10}],
    #     'Fear': [{20}, {1, 4, 5, 25}],
    #     'Disgust': [{7,24},{1, 4, 9, 10, 14, 15, 17, 25}],
    #     'Sadness': [{43},{1, 4, 14, 15, 17, 43}]
    # }
    emotion_aus = {
        'Happiness': [{6, 12}, {1, 14, 26, 27}],
        'Surprise': [{2}, {1, 5, 25, 26, 27}],
        'Anger': [{16, 22, 23}, {4, 9, 10}],
        'Fear': [{20}, {1, 4, 5, 25}],
        'Disgust': [{7, 24}, {1, 4, 9, 10, 14, 15, 17, 25}],
        'Sadness': [{1}, {4, 14, 15, 17}]
    }

    # 将输入的AU转换为集合
    input_aus = set(aus+1)
    emotion_main = []
    emotion_sub = []
    emotions = []
    # 检查输入的AU是否与任何情绪的AU集合匹配
    for emotion, action_units in emotion_aus.items():
        # 检查独有的AU集合
        emotions.append(emotion)
        emotion_main.append(len(input_aus&action_units[0])/len(input_aus|action_units[0]))
        emotion_sub.append(len(input_aus&action_units[1])/len(input_aus|action_units[1]))
        # if input_aus.intersection(action_units[0]):
        #     # 如果输入的AU与独有的AU有交集，进一步检查共享的AU
        #     if input_aus.issubset(action_units[0].union(action_units[1])):
        #         return emotion
        #     # 如果独有的AU不满足，检查共享的AU集合
        # elif input_aus.issubset(action_units[1]):
        #     return emotion
    max_score = 0
    for i in range(len(emotion_main)):
        score = emotion_main[i] * 2 + emotion_sub[i]
        if score > 0.4:
            if max_score < score:
                max_score = score
                emotion_i = emotions[i]
    if max_score == 0:
        return 'wo emotion'
    return emotion_i


# 示例：对一个包含AU的frame进行分类
frame_aus = np.array([1, 6, 12])  # 假设这是某个frame检测到的AU列表
expression = classify_expression(frame_aus)
if expression:
    print(f"The classified expression is: {expression}")
else:
    print("No specific expression classified, the AUs may not match any specific emotion or require further analysis.")
# 示例：使用函数将AU编号转换为面部区域
au_array = np.array([1, 4, 17, 26, 38])
facial_regions = au_to_facial_region(au_array)

print(facial_regions)