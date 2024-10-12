import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from datetime import datetime
from base_model import Net  # 导入Net模型

# 定义数据加载器函数
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("./data_train", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 定义模型评估函数:为模型打分
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():  # 评估时不需要梯度计算
        for (x, y) in test_data:
            # 处理输入并得到输出
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                # 获取预测值，并与实际标签比较
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    # 返回准确率
    return n_correct / n_total




# 如果该文件作为主程序执行，则调用 main 函数
if __name__ == "__main__":
    # 加载训练和测试数据
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    # 初始模型的准确率
    
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 训练模型
    # for epoch in range(2):
    epoch=0
    true_rate=evaluate(test_data, net)
    
    print(f"训练迭代\t{epoch}次\t准确率:{true_rate}")
    while True:
        epoch += 1
        for (x, y) in train_data:
            net.zero_grad()  # 梯度清零
            # 前向传播
            output = net.forward(x.view(-1, 28*28))
            # 计算损失
            loss = torch.nn.functional.nll_loss(output, y)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        # 每个epoch后打印测试集上的准确率
        true_rate=evaluate(test_data, net)
        print(f"训练迭代\t{epoch}次\t准确率:{true_rate}")
        if true_rate >= 0.975:
            print("-"*50)
            print("模型已达标,最终准确率:",true_rate)
            # 保存模型的state_dict
            nowData=datetime.now().strftime("%Y%m%d%H%M%S")
            model_name=f"./model/numbermodel_{epoch}_{true_rate}_{nowData}.pth"
            torch.save(net.state_dict(),model_name)
            break;