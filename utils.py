import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class ModelUtils:
    """
    模型工具类，提供数据加载和评估方法，方便复用。
    """

    

    @staticmethod
    def get_data_loader(is_train,batch_size):
        """
        定义数据加载器函数
        获取 MNIST 数据集的并转换为张量类型
        
        参数：
        is_train (bool): 如果为 True，加载训练数据集；如果为 False，加载测试数据集。

        返回：
        DataLoader: 返回一个数据加载器，用于批量加载数据。
        """
        # 转换为张量是深度学习和机器学习中最基本的数据结构之一(机器学习处理张量数据)
        to_tensor = transforms.Compose([transforms.ToTensor()])
        # 从 MNIST 数据集中加载数据
        # "./data_train": 数据存储路径
        # is_train: 是否加载训练集（如果为 True 则加载训练集，False 则加载测试集）
        # transform=to_tensor: 将数据转换为张量
        # download=True: 如果数据集不存在，则自动下载
        data_set = MNIST("./data_train", is_train, transform=to_tensor, download=True)
        # 创建 DataLoader，
        # 设置批大小为 15，
        # shuffle=True 表示每个 epoch 开始前随机打乱数据
        return DataLoader(data_set, batch_size, shuffle=True)
    @staticmethod
    def evaluate(data_test, net):
        """
        定义模型评估函数，计算模型在测试集上的准确率

        参数：
        data_test (DataLoader): 测试数据加载器，包含测试样本及其标签。
        net (torch.nn.Module): 已训练的神经网络模型。

        返回：
        float: 模型在测试集上的准确率。
        """
        num_true = 0  # 记录预测正确的样本数量
        num_total = 0  # 记录总样本数量
        with torch.no_grad():  # 禁用梯度计算，节省内存和加速计算
            # 遍历测试集，elements 是输入图像数据，labels 是对应标签
            for (elements, labels) in data_test:
                # 将图像展平为一维向量，并通过模型得到预测结果
                outputs = net(elements.view(-1, 28*28))

                predictions = torch.argmax(outputs, dim=1)  # 预测结果
                num_true += (predictions == labels).sum().item()  # 统计正确预测的数量
                num_total += labels.size(0)  # 统计总样本数量
        
        # 返回测试集上的准确率
        return num_true / num_total


