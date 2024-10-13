#未训练的标准神经网络模型定义
import torch  # 导入 PyTorch 库

class Net(torch.nn.Module):
    # 定义一个神经网络模型 Net，继承自 torch.nn.Module
    def __init__(self):
        """初始化父类
        并定义神经网络各层输入输出维度(结构)
        self: 当前实例
        super(Net, self).__init__()：调用父类的构造函数，初始化父类，
        """
        super(Net, self).__init__()
        
        # 定义第一个全连接层(torch.nn.Linear)，self=this
        # 将输入的图像特征28*28=784通过一个线性变换映射输出到64维的空间(第一层)
        self.fc1 = torch.nn.Linear(28*28, 64)  
        
        # 定义第二个全连接层，将输入64维输出64维(第二层)
        self.fc2 = torch.nn.Linear(64, 64)      
        
        # 定义第二个全连接层，将输入64维输出64维(第三层)
        self.fc3 = torch.nn.Linear(64, 64)      
        
        # 定义第四个全连接层，将64维特征映射到10维，输出10个类别的预测值(结果)
        self.fc4 = torch.nn.Linear(64, 10)      

    def forward(self, x):
        """定义前向(前进)传播函数:1层(输入)->2层->3层,4层(输出)的规则
        x: 输入数据
        self: 当前实例
        
        前三层使用realu激活函数,原因:
            非线性变换:f(x)=max(0,x)。它将输入值小于 0 的部分输出为 0，而大于 0 的部分保持不变。这种非线性变换允许神经网络学习复杂的特征和模式。
            
            隐藏层的作用：在隐藏层使用 ReLU 是为了引入非线性，使得每一层的输出能够捕捉到输入数据中的复杂特征。多层的 ReLU 激活函数组合可以让网络学习更复杂的函数映射。
            
            避免梯度消失：ReLU 激活函数在正区域的梯度为 1，这可以有效地缓解传统激活函数（如 sigmoid 或 tanh）在深层网络中常见的梯度消失问题，从而加速训练过程。
            
        输出层使用log_softmax 激活函数
        输出层的概率分布：log_softmax 将最后一层的线性输出（logits）转换为对数概率。其作用是计算每个类别的相对可能性，适用于多类分类任务。
        对数概率：通过使用log_softmax，可以直接获得对数概率，这在后续计算损失时（如交叉熵损失）非常有用。使用对数概率可以提高数值稳定性，并避免在计算中出现非常小的概率值。
        
        ReLU：用于隐藏层，提供非线性变换，帮助网络学习复杂的特征。
        log_softmax：用于输出层，将最后一层的输出转换为对数概率，以便进行有效的损失计算和分类。
        """
        
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
