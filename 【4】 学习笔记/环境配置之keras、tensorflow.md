#### 一、安装环境

windows 10

python 3.7.6-64bit

Anaconda3 2020.02（https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe）

Visual Studio Code

#### 二、**TensorFlow，keras具体安装步骤**

1、版本对应原则（https://docs.floydhub.com/guides/environments/）

```
conda remove tensorflow
conda uninstall tensorflow
conda install tensorflow==1.15.0
conda install keras==2.3.1
```

conda安装失败，换为pip重装：

```
pip uninstall tensorflow
pip install tensorflow==1.15.0
pip install keras==2.3.1
```

```
python
import tensorflow as tf
tf.__version__
```

```
python
import keras
keras.__version__
```

![image-20210124144735752](C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210124144735752.png)![image-20210124144801801](C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210124144801801.png)

直到查出版本号，才算安装成功。否则均失败。

2、Anaconda换源

![img](file:///D:\Users\86177\Documents\Tencent Files\499128175\Image\C2C\`%}A{QFP3SOXPOH~[304VXM.png)

ADD：

https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2

https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch

https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo

https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
然后点updatechannals

3、pip下载太慢，打开cmd 把下面的代码复制进去 回车

c:
cd c:%HOMEPATH%
mkdir pip
cd pip
(echo [global]
echo timeout = 6000
echo index-url = https://pypi.tuna.tsinghua.edu.cn/simple
echo trusted-host =pypi.tuna.tsinghua.edu.cn)>pip.ini

#### 三、LLE算法实验

1、装好tensorflow1.15之后，把数据集文件放到 Anaconda主目录\Lib\site-packages\tensorflow_core\examples

<img src="file:///D:\Users\86177\Documents\Tencent Files\499128175\Image\C2C\}{@VC(Z2(5V]ACG1ZO@%)HK.png" alt="img" style="zoom:67%;" />

<img src="C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210124145348874.png" alt="image-20210124145348874" style="zoom:67%;" />

2、实验结果

<img src="C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210124145512127.png" alt="image-20210124145512127" style="zoom:67%;" />

#### 四、总结

1、tensorflow、keras版本要对应，tensorflow2与1差别很大，tensorflow2时2019年发布。

2、数据集下载

#### 五、Pytorch安装

1、版本匹配

![img](file:///C:\Users\86177\AppData\Roaming\Tencent\QQTempSys\%W@GJ$ACOF(TYDYECOKVDYB.png)https://pytorch.org/get-started/locally/

2、无cuda

![image-20210310205516852](C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210310205516852.png)

3、在cmd运行这句命令

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

![image-20210310205610144](C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210310205610144.png)

4、运行报错

![image-20210310205655932](C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210310205655932.png)

解决办法：

![image-20210310205718003](C:\Users\86177\AppData\Roaming\Typora\typora-user-images\image-20210310205718003.png)

```
conda install pytorch==1.5 torchvision==0.6 torchaudio cpuonly -c pytorch
```

