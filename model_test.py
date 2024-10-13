import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model import Net  # 导入Net模型
from utils import ModelUtils
# 实例化并加载训练好的模型
net = Net()
model_path = './model/numbermodel_7_0.976_20241012203328.pth'
net.load_state_dict(torch.load(model_path))
net.eval()  # 设置模型为评估模式


test_loader=ModelUtils.get_data_loader(False,25)
print(test_loader)
# 获取一批测试数据
images, labels = next(iter(test_loader))
# # 使用模型进行预测
with torch.no_grad():
    outputs = net(images.view(-1, 28*28))  # 将图片展平
    predictions = torch.argmax(outputs, dim=1)
print(predictions)
# # 创建子图，2行2列
# fig, axs = plt.subplots(5, 5)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(images[i].view(28,28))
    plt.title(f"{predictions[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()