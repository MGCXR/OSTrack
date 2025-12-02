import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import (
    get_default_qconfig,
    fuse_modules,
    prepare_fx,
    convert_fx
)

# ======================================
# 1. 加载浮点模型（你可以换成自己的）
# ======================================
model_file='./convert/test_mf/models/Test_ep0300.pth'
model=torch.load(model_file,map_location='cpu',weights_only=False) 
model.eval()

# ======================================
# 2. 模块融合（根据你的模型做调整）
# ResNet18 只举例第一层，其它层官方已融合好
# ======================================
fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)

# ======================================
# 3. 设置量化配置（CPU x86 用 fbgemm）
# ======================================
qconfig = get_default_qconfig("fbgemm")
template = torch.randn(1, 3, 128, 128) # 用于 FX tracing
search = torch.randn(1, 3, 256, 256)

prepared_model = prepare_fx(model, {"template": template, "search": search,'template_event':template,'search_event':search})

# ======================================
# 4. 校准（不训练，只前向推理）
# 用少量真实数据即可（几十张即可）
# ======================================
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

# 示例：使用 CIFAR 10 校准，你可以换成自己的数据
dataset = torchvision.datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform
)
calibration_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

def calibrate(model, loader, num_batches=10):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images)
            if i + 1 >= num_batches:
                break

print("开始校准...")
calibrate(prepared_model, calibration_loader)

# ======================================
# 5. 转换为量化模型（int8）
# ======================================
quantized_model = convert_fx(prepared_model)
quantized_model.eval()

# ======================================
# 6. 测试量化模型推理
# ======================================
test_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = quantized_model(test_input)
print("量化模型输出大小:", out.shape)

# ======================================
# 7. 保存模型
# ======================================
torch.save(quantized_model, 'quantized_model.pth')
print("量化模型已保存为 quantized_model.pth")