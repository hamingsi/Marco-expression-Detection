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

# 创建一个示例输入
dummy_input = torch.randn(1, 3,224,224)

# 导出ONNX模型
torch.onnx.export(net, dummy_input, "simple_model.onnx", 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})

print("ONNX模型已导出")

# 验证ONNX模型
onnx_model = onnx.load("simple_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX模型验证通过")