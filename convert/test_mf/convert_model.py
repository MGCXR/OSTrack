import numpy as np
if not hasattr(np, "alltrue"):
    np.alltrue = np.all
import sys
import onnx
import onnxruntime as ort
import random
import torch
from convert.units.samples_common import convert_model
from torchvision import transforms
import lyngor as lyn
# import onnx
from collections import Counter
# 定义常数变量
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

template=torch.randn(1,3,128,128) 
template = (template - template.min()) / (template.max() - template.min())   
search=torch.randn(1,3,256,256)
search = (search - search.min()) / (search.max() - search.min())   
template_event=torch.randn(1,3,128,128)
template_event = (template_event - template_event.min()) / (template_event.max() - template_event.min())   
search_event=torch.randn(1,3,256,256)
search_event = (search_event - search_event.min()) / (search_event.max() - search_event.min())   

template_np = template.numpy()
search_np = search.numpy()
template_event_np = template_event.numpy()
search_event_np = search_event.numpy()

inputs_dict={'template': template_np, 'search': search_np, 'template_event': template_event_np, 'search_event': search_event_np}
# np.save('./convert/test_mf/inputs_dict.dat', inputs_dict)


# model_file = 'convert/test_mf/Test_ep0300.onnx'
model_file = 'convert/test_mf/models/output.onnx'
target = 'apu'
dtype = 'float32'

# # # 转换模型，获取引擎
# r_engine = convert_model(model_file, 
#                          inputs_dict={'template':(1,3,128,128),'search':(1,3,256,256),'template_event':(1,3,128,128),'search_event':(1,3,256,256)},
#                          target=target,
#                          path='./convert/test_mf/tmp_net/onnx/', 
#                          model_type='ONNX', 
#                          build_mode="auto",
#                          profiler=True,
#                          serialize=False)


# model = onnx.load(model_file)

# ops = [node.op_type for node in model.graph.node]
# print("算子数量统计：")
# for k, v in Counter(ops).items():
#     print(f"{k:20s}: {v}")


# model_onnx = onnx.load(model_file)
model_onnx = ort.InferenceSession(model_file)

output_onnx = model_onnx.run(None, {'template': template_np, 'search': search_np, 'template_event': template_event_np, 'search_event': search_event_np})
# np.save('./convert/test_mf/output_onnx.dat', output_onnx)
# r_engine = lyn.load(path='./convert/test_mf/tmp_net/onnx/Net_0', device=0)

# @lyn.PerfLogger(config='./convert/test_mf/tmp_net/onnx/Net_0/apu_0/profiler/debug_chip0.json', output="./convert/test_mf/tmp_net/output")
# def run_case():
#     r_engine.run(template=template_np, search=search_np, template_event=template_event_np, search_event=search_event_np)
# run_case()

# r_engine.run(template=template_np, search=search_np, template_event=template_event_np, search_event=search_event_np)
# output = r_engine.get_output()

# print('-------------------------------------------------------------------------\n',output_onnx,
#       '\n','-------------------------------------------------------------------------\n',output)

















# # -----------------------------------------------------------------------------
# model_file = 'convert/test_mf/Test_ep0300.pth'
# # model_file = 'convert/test_mf/Test_ep0300_scripted.pth'
# target = 'apu'
# dtype = 'float32'






# model_torch = torch.load(model_file, map_location='cpu',weights_only=False)
# model_torch.eval()
# with torch.no_grad():
#     output_torch = model_torch(template, search, template_event, search_event)


# # r_engine = convert_model(model_torch, 
# #                          inputs_dict={'template':(1,3,128,128),'search':(1,3,256,256),'template_event':(1,3,128,128),'search_event':(1,3,256,256)},
# #                          target=target,
# #                          path='./convert/test_mf/tmp_net/pytorch/', 
# #                          model_type='PyTorch', 
# #                          build_mode="auto",
# #                          profiler=False)


# r_engine = lyn.load(path='./convert/test_mf/tmp_net/pytorch/Net_0/', device=0)

# # r_engine.set_input('template', template_np)
# # r_engine.set_input('search', search_np)
# # r_engine.set_input('template_event', template_event_np)
# # r_engine.set_input('search_event', search_event_np)
# r_engine.run(template=template_np, search=search_np, template_event=template_event_np, search_event=search_event_np)
# output = r_engine.get_output()
# # print("输出结果：", output[0])
# print(output_torch['score_map'],'------------------------------------------------\n',output_onnx[0])
# print(output_torch[0],'\n',output_onnx[0])
# print(output_torch['pred_boxes'],'\n',output)