## 一、《DeepFashion论文》讲解

<img src="C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20201122205136691.png" alt="image-20201122205136691" style="zoom: 80%;" />

全局、局部

<img src="C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20201122210518659.png" alt="image-20201122210518659" style="zoom:50%;" />

fc全连接

http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html《DeepFashion论文》

## 二、tensorflow遇到的问题

### 问题一：Failed to load the native TensorFlow runtime.

方法：https://blog.csdn.net/herr_kun/article/details/80448861?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242

背景及环境介绍：重装了系统，发现使用TensorFlow时出现了Failed to load the native TensorFlow runtime. 

目前环境：win10 、anaconda 、python3.6 、TensorFlow—CPU

罗列一下博客以及Stack Overflow上的解答：

部分参考来自[这里](https://blog.csdn.net/Kexiii/article/details/77990459)，但是上述均没有解决我的问题。经过一系列的摸索得出解决方案：

1、先使用conda upgrade --all 升级所有的包

2、使用Python -m pip install --upgrade pip 或者conda upgrade pip升级pip工具

3、使用conda install tensorflow 安装（因为pip install TensorFlow安装成功，但是import一直会出现上述问题，虽然TF不对conda上的TF进行维护，但是目前TF1.2版本的已经够用了）

​	 小插曲

**解决办法：**https://blog.csdn.net/weixin_44102198/article/details/109189361
创建一个兼容的python的环境即可：
\#建立新环境（我创建的是python3.7`#建立新环境)
$: conda create -n python3.7
: conda activate python3.7 (python3.5)

4、完成上一步之后就有了TensorFlow的backend,在使用pip install keras 或者conda install keras均可

上述问题解决

### 问题二：tensorflow安装不成功

python3.8环境太高，需要3.5-3.7版本，重装Anaconda

conda install tensorflow即可成功

### 问题三：KeyError: 'accuracy'、KeyError: 'val_acc'、KeyError: 'acc'等报错信息的解决方法

https://blog.csdn.net/weixin_43051346/article/details/103647390

报错信息之KeyError: ‘accuracy’

之所以会报错，是因为keras库老版本中的参数不是accuracy，而是acc，将参数accuracy替换为acc

## 三、卷积神经网络

### 3.1 华为云平台https://storage.huaweicloud.com/

创建桶

wget命令不成功