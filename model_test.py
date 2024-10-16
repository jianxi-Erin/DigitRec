import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model import Net  # 导入Net模型
from utils import ModelUtils
# 使用自定义模型预测minst数据集(有标签)

# 实例化并加载训练好的模型
net = Net()
model_path = './model/numbermodel_7_0.976_20241012203328.pth'
net.load_state_dict(torch.load(model_path))
net.eval()  # 设置模型为评估模式

test_loader=ModelUtils.get_data_loader(False,50)

# 获取一批测试数据
images, labels = next(iter(test_loader))
print("图片张数:",images.size())
print(images.shape)
# # 使用模型进行预测
with torch.no_grad():
    outputs = net(images.view(-1, 28*28))  # 将图片展平
    predictions = torch.argmax(outputs, dim=1)
    # 当前批的准确率
    correct_predictions = (predictions == labels).sum().item()  # 正确预测的数量
    total_samples = labels.size(0)  # 总样本数量
    true_rate = correct_predictions / total_samples  # 当前批次的准确率

print("预测:",predictions)
print("实际:", labels)
print("准确率:",true_rate)

# # 创建子图
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.imshow(images[i].view(28,28),cmap="gray")
    plt.title(f"{predictions[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()