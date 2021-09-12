# !/usr/bin/python
# coding: utf8
# @Time    :
# @Author  :
# @Email   :
# @Software: PyCharm

from torch import nn
import torch.nn.functional as F

'''
    原始输入样本的大小：32 x 32 x 3
    第一次卷积：使用6个大小为5 x 5的卷积核，故卷积核的规模为(5 x 5) x 6；卷积操作的stride参数默认值为1 x 1，32 - 5 + 1 = 28，并且使用ReLU对第一次卷积后的结果进行非线性处理，输出大小为28 x 28 x 6；
    第一次卷积后池化：kernel_size为2 x 2，输出大小变为14 x 14 x 6；
    第二次卷积：使用16个卷积核，故卷积核的规模为(5 x 5 x 6) x 16；使用ReLU对第二次卷积后的结果进行非线性处理，14 - 5 + 1 = 10，故输出大小为10 x 10 x 16；
    第二次卷积后池化：kernel_size同样为2 x 2，输出大小变为5 x 5 x 16；
    第一次全连接：将上一步得到的结果铺平成一维向量形式，5 x 5 x 16 = 400，即输入大小为400 x 1，W大小为120 x 400，输出大小为120 x 1；
    第二次全连接，W大小为84 x 120，输入大小为120 x 1，输出大小为84 x 1；
    第三次全连接：W大小为10 x 84，输入大小为84 x 1，输出大小为10 x 1，即分别预测为10类的概率值。

    平坦层：400个神经元
    到隐藏层一：120个神经元
    到隐藏层二：84个神经元
    到隐藏层三：10个神经元
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #  conv1层，输入的灰度图，所以 in_channels=1, out_channels=6 说明使用了6个滤波器/卷积核，
        # kernel_size=5卷积核大小5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # conv2层， 输入通道in_channels 要等于上一层的 out_channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # an affine operarion: y = Wx + b
        # 全连接层fc1,因为32x32图像输入到fc1层时候，feature map为： 5x5x16
        # 因此，全连接层的输入特征维度为16*5*5，  因为上一层conv2的out_channels=16
        # out_features=84,输出维度为84，代表该层为84个神经元
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 特征图转换为一个１维的向量
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），
    # 比如咱们上面的x是5*5*16的张量，那么它的特征总量就是400。
    def num_flat_features(self, x):
        # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，
        # 那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

