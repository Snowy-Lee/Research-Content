### 一、卷积神经网络实验设计

#### 1.1 实验介绍

​		网络原理
​		卷积神经网络（CNN）是深度学习算法应用最成功的领域。CNN是一类包含卷积计算且具有深度结构的前馈神经网络。相较于前馈神经网络，卷积神经网络考虑了数据的“形状”、模拟了生物视觉感受野。卷积神经网络中包括输入层、隐含层和输出层。其中，隐含层包括了卷积层、池化层和激活函数，其中卷积层可以被舍去。

#### 1.2 实验目的

​		通过实验了解卷积神经网络实现的细节，学会搭建卷积神经网络，了解例如dropout的正则方法及其效果，掌握卷积神经网络应用在分类的流程。

### 二、实验设计

#### 2.1 实验数据

​	本实验使用Fashion-MNIST 数据集,它是一个替代MNIST手写数字集的图像数据集，涵盖了10种类别共7万个不同商品的正面图片。Fashion-MNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致，包含60,000个用于训练的示例和10,000个用于测试的示例，图像的大小为固定的28×28像素。

#### 2.2 实验环境

​	本次实验在华为云 MindActs.平台上基于MindSpoxe进行编程测试。

### 三、实验流程

<img src="C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20201206184116822.png" alt="image-20201206184116822" style="zoom:67%;" />

(1) 导入实验环境:将实验代码中所需的python 包用import命令导入。

(2) 数据准备:在代码中添加下载数据集和解压的语句。

(3) 训练模型:在MindArts开发环境中，新建Notebook，上传代码文件，在Jupyter中运行。

(4) 可视化结果:对预测的运行结果进行可视化。

### 四、实验步骤

​		略。

