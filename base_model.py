#未训练的标准神经网络模型定义
import torch  # 导入 PyTorch 库

class Net(torch.nn.Module):
    # 定义一个神经网络模型 Net，继承自 torch.nn.Module
    def __init__(self):
        # 初始化父类
        super(Net, self).__init__()
        
        # 定义第一个全连接层，将输入的28*28=784维特征映射到64维
        self.fc1 = torch.nn.Linear(28*28, 64)  
        
        # 定义第二个全连接层，将64维特征映射到64维
        self.fc2 = torch.nn.Linear(64, 64)      
        
        # 定义第三个全连接层，将64维特征映射到64维
        self.fc3 = torch.nn.Linear(64, 64)      
        
        # 定义第四个全连接层，将64维特征映射到10维，输出10个类别的预测值
        self.fc4 = torch.nn.Linear(64, 10)      

    def forward(self, x):
        # 定义前向传播函数
        # 对输入数据 x 进行第一层的线性变换并应用 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc1(x))  
        
        # 对第一层的输出进行第二层的线性变换并应用 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc2(x))  
        
        # 对第二层的输出进行第三层的线性变换并应用 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc3(x))  
        
        # 对第三层的输出进行第四层的线性变换并应用 log_softmax 激活函数
        # 这里的 log_softmax 将输出转换为对数概率，dim=1 表示按行计算
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1) 
        
        # 返回最终的输出结果
        return x  
