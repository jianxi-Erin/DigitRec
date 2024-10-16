import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model import Net  # 导入Net模型
from utils import ModelUtils
# 使用自定义数据集合(无标配)预测


# 设置 Matplotlib 使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# 实例化并加载训练好的模型
net = Net()
model_path = './model/numbermodel_7_0.976_20241012203328.pth'
net.load_state_dict(torch.load(model_path))
net.eval()  # 设置模型为评估模式

# 定义自定义测试集的转换
transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图像
    transforms.Resize((28, 28)),  # 调整大小为28x28
    transforms.ToTensor(),  # 转换为Tensor
])

# 创建自定义测试集的数据加载器
custom_test_dataset = datasets.ImageFolder(root="./data_test/custom_data_sets/", transform=transform)
test_loader = DataLoader(custom_test_dataset, batch_size=50, shuffle=False)

# 使用模型进行预测
epoch=0
for images, labels in test_loader:
    output = f"第{epoch}批数据"
    print(f"{output:-^25}")
    print("输入Image:", images.size())
    print("输入label:",labels)
    with torch.no_grad():
        outputs = net(images.view(-1, 28 * 28))  # 将图片展平
        predictions = torch.argmax(outputs, dim=1)
    print("预测输出label:", predictions)

    # 创建子图
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{epoch}批手写数字预测", fontsize=16)
    
    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.imshow(images[i].view(28,28),cmap="gray")
        plt.title(f"{predictions[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    epoch+=1
