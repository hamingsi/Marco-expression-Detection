import torch
import torch.nn as nn
import onnx
import onnxruntime
import time
import numpy as np
from conf import get_config, set_logger, set_outdir, set_env

conf = get_config()
conf.evaluate = True
set_env(conf)
# generate outdir name
set_outdir(conf)
# Set the logger
set_logger(conf)
device = torch.device("mps")
if conf.stage == 1:
    from model.ANFL import MEFARG

    net = MEFARG(num_main_classes=27, num_sub_classes=14, backbone=conf.arc,
                    neighbor_num=4, metric=conf.metric)
# 创建模型实例
# net = net.to(device)
net.eval()  # 设置为评估模式

# 创建ONNX运行时会话
ort_session = onnxruntime.InferenceSession("simple_model.onnx", providers=['CoreMLExecutionProvider'])
ort_session2 = onnxruntime.InferenceSession("simple_model.onnx", providers=['CPUExecutionProvider'])

# 准备测试数据
test_input = np.random.randn(1,3,224,224).astype(np.float32)

# PyTorch模型推理
def run_pytorch(input_data):
    # net.to(device)
    with torch.no_grad():
        output = net(torch.from_numpy(input_data))
    return output.numpy()

# ONNX模型推理
def run_onnx(input_data):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def run_onnx2(input_data):
    ort_inputs = {ort_session2.get_inputs()[0].name: input_data}
    ort_outs = ort_session2.run(None, ort_inputs)
    return ort_outs[0]

# 测试推理速度
def test_speed(run_func, input_data, name):
    start = time.time()
    for _ in range(100):  # 运行100次以获得更准确的时间
        run_func(input_data)
    end = time.time()
    print(f"{name} 推理时间: {end - start:.4f} 秒")

# 运行速度测试
test_speed(run_pytorch, test_input, "PyTorch")
test_speed(run_onnx2, test_input, "ONNX CPU")
test_speed(run_onnx, test_input, "ONNX CoreML")

# 验证输出结果是否一致
pytorch_output = run_pytorch(test_input)
onnx_output = run_onnx(test_input)
print(onnxruntime.get_available_providers())
np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-03, atol=1e-05)
print("PyTorch和ONNX输出结果一致")