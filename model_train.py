import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from datetime import datetime
from base_model import Net  # 导入Net模型
from utils import ModelUtils #导入模型工具类

# 使用自定义神经网络基础模型通过minst训练集(有标签)训练模型,并通过测试集评估模型准确率\



# 如果该文件作为主程序执行，则调用 main 函数
if __name__ == "__main__":
    # 1.实例化基础模型
    net = Net()
    
    # 2. 加载训练和测试数据
    train_data = ModelUtils.get_data_loader(True,50)
    test_data = ModelUtils.get_data_loader(False,50)
    
    
    # 3. 定义模型参数优化器 Adam,lr为参数更新步长(太小会速度慢,太大可能会错过最优参数)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 4 .训练模型
    
    epoch=0 #记录迭代次数
    true_rate=ModelUtils.evaluate(test_data, net) #计算模型准确率
    print(f"训练迭代\t{epoch}次\t准确率:{true_rate}") #打印初始迭代次数和准确率
    #   循环训练
    while True:
        epoch += 1
        for (element, label) in train_data:
            # 梯度清零:每次执行前向传播和反向传播之前，都需要将模型的梯度清零。否则，PyTorch 会在现有的梯度基础上累加新计算的梯度，影响训练结果。
            net.zero_grad()  

            # 前向传播会自动调用forward()方法:输入层->隐藏层->输出层,得到预测结果output
            output = net(element.view(-1, 28*28))
            
            # 计算损失:使用负对数似然损失函数(nll_loss)来计算模型的output与label之间的差异（损失值）
            # 损失函数的值越小，表示模型的预测性能越好。
            loss = torch.nn.functional.nll_loss(output, label)
            
            # 反向传播:通过链式法则计算,指明了如何调整参数以减少损失。
            loss.backward()
            
            # 更新参数:使用Adam优化器更新模型的参数，基于前面计算出的梯度对模型进行调整，从而使损失函数值更小。
            optimizer.step()
        
        # 每个epoch后使用测试集计算准确率
        true_rate=ModelUtils.evaluate(test_data, net)
        print(f"训练迭代\t{epoch}次\t准确率:{true_rate}")
        
         # 当准确率高于0.975,保存模型的,退出训练
        if true_rate >= 0.975:
            
            print("-"*50)
            print("模型已达标,最终准确率:",true_rate)
           
            nowData=datetime.now().strftime("%Y%m%d%H%M%S")
            model_name=f"./model/numbermodel_{epoch}_{true_rate}_{nowData}.pth"
            torch.save(net.state_dict(),model_name)
            
            break;