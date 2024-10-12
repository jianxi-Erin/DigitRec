import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model import Net  # 导入Net模型
# 实例化并加载训练好的模型
net = Net()
model_path = './model/numbermodel_7_0.976_20241012203328.pth'
net.load_state_dict(torch.load(model_path))
net.eval()  # 设置为评估模式

# 加载MNIST测试数据
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data_test', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

# 获取一批测试数据
images, labels = next(iter(test_loader))

# 使用模型进行预测
with torch.no_grad():
    outputs = net(images.view(-1, 28*28))  # 将图片展平
    predictions = torch.argmax(outputs, dim=1)

# 创建子图，2行2列
fig, axs = plt.subplots(5, 5)

# 显示图片及预测结果
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(images[i*2 + j].view(28, 28), cmap='gray')  # 显示图片
        axs[i, j].set_title(f"R: {predictions[i*2 + j].item()}")  # 显示预测结果
        axs[i, j].axis('off')  # 隐藏坐标轴

# 调整布局并显示
plt.tight_layout()
plt.show()
