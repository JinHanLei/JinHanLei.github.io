---
title: 深度学习环境配置：Anaconda、CUDA、PyTorch安装和那些坑儿

date: 2023-05-27 17:00:00 +0800

categories: [Deep Learning]

tags: [deep learning]

comments: false

mermaid: true

description: Anaconda下载及安装，CUDA安装，PyTorch安装
---

工欲善其事，必先利其器。在使用Python开展深度学习的研究之前，绕不开的一步就是环境安装。本章将介绍Python的管理工具Anaconda、NVIDIA驱动和CUDA的安装、深度学习框架PyTorch的安装和那些坑儿。这几者的关系如下图：

```mermaid
graph LR
A[数据] -->B(PyTorch)
    B -->|Anaconda| C(Python)
	C --> D(CUDA) --> E(NVIDIA驱动) --> F(显卡)
```

我们把数据交给PyTorch，Python进行代码解释，调用CUDA的深度学习工具，最后交给驱动去用显卡，而Anaconda的作用在于Python的便捷管理。环环相扣，缺一不可，让我们开始安装之旅！

## Anaconda安装和使用

### 安装Anaconda

[Anaconda](https://www.anaconda.com/download/)是一个对Python包和环境进行管理的工具。主要功能有：

- 帮你预装好大部分常用的Python包，如Jupyter Notebook；
- 快速创建、克隆、删除虚拟环境。当一个程序需要使用Python 2.7版本，而另一个程序需要使用Python 3.9版本，在Anaconda中只需要新建一个虚拟环境即可。

Anaconda的安装十分简单，只需：

1. 从[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)下载安装包。点击时间最新的，对应系统的就行。不想预装那么多包的可以装[miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)。如果需要装固定版本，对应版本在[这里](https://docs.anaconda.com/free/anaconda/reference/packages/oldpkglists/)；
2. 不断点击下一步。我勾选了”Add Anaconda3 to the system PATH“也没事，如果求稳的话，可以不勾选，手动配置Anaconda下环境变量。

这里有个坑在于：如果计算机的**用户名**是**中文，**会报各种错误，建议修改或者重装系统。如果用户名是英文或者数字，还报错，建议换个较早的版本安装，如Anaconda3-2021.04版，python版本为3.6。

### 使用Anaconda

装好后Windows在开始菜单找到<kbd>Anaconda3(64-bit)</kbd>，点击<kbd>Anaconda prompt</kbd>；Linux重启一下命令行终端，就可以愉快使用了，环境配置和安装包都在这里进行。

Anaconda的使用也相当方便。常用的就是操作虚拟环境。虚拟环境是各自完全独立的python包空间，可以安装不同版本的python及不同版本的包，Anaconda可以快速操作，命令如下，`<env_name>`替换成你想要的环境名：

```shell
# 创建
conda create -n <env_name> python=3.6
# 克隆一个一模一样的A环境
conda create -n <env_name> --clone A
# 进入虚拟环境
conda activate <env_name>
# 退出当前环境
conda deactivate
# 查看已有的虚拟环境
conda env list
# 删除环境
conda env remove -n <env_name>
```

在各个环境里可以`pip install`，也可以`conda install`来装包。

并且环境可以在Pycharm中引入使用，进入Pycharm点击<kbd>Edit</kbd>-<kbd>Settings</kbd>-<kbd>Python Interpreter</kbd>-<kbd>Add Interpreter</kbd>-<kbd>Conda Environment</kbd>，会帮你自动找到conda和虚拟环境，添加即可。

### 更换conda和pip源

国内访问conda和pip可能会比较慢，装包很不方便，我们先更换成清华源。

清华镜像站都提供了教程：

- [conda更换源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)：其中提到的`.condarc`在C盘-用户-`用户名`文件夹中；
- [pip更换源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)：`设为默认`节中的两行命令就搞定。

### Jupyter Notebook添加虚拟环境

Jupyter Notebook是以网页的形式打开，可以在网页页面中直接编写代码和运行代码，代码的运行结果也会直接在代码块下显示的程序，主打一个方便。

Anaconda默认是装好jupyter的，只需运行：

```shell
jupyter notebook
```

但是当Anaconda的虚拟环境创建完成，在jupyter上是没有的。需要手动添加，在虚拟环境中执行命令：

```shell
pip install --user ipykernel
python -m ipykernel install --user --name=<env_name>
```

这样在新建`.ipynb`文件或者在打开的`.ipynb`文件右上角，就可以选择虚拟环境了。删除虚拟环境只需：

```shell
# 查看当前jupyter环境列表
jupyter kernelspec list
# 删除
jupyter kernelspec uninstall <env_name>
```

想要在后台启动jupyter，不想要命令行窗口一直开着，可执行：

```shell
nohup jupyter notebook --ip 0.0.0.0 --port 8012 --allow-root > jupyter.log 2>&1 &
# 查看
ps -aux | grep jupyter
```

在计算机开启时会一直在后台运行，适合服务器上用，关闭计算机或者重启之后就需要再次执行了喔。

## 安装NVIDIA驱动和CUDA

NVIDIA显卡是做深度学习的基础设施。有NVIDIA驱动才能调度显卡，有CUDA才能在显卡上做深度学习。

### 安装NVIDIA驱动

首先检查自己的计算机是否有N卡，有如下方案：

1. 观察外机是否贴着绿油油的NVIDIA标；
2. 开始菜单查看是否预装了NVIDIA相关软件；
3. 命令行运行dxdiag，弹出窗口，查看<kbd>显示</kbd>下是否有NVIDIA设备；
4. 询问购机处。

是N卡但没装驱动，请前往[NVIDIA官网](https://www.nvidia.cn/Download/index.aspx?lang=cn)选择自己显卡的版本进行安装，驱动越新越好。下载后双击，只需不断下一步就安装好了。装好后在命令行工具验证：

```shell
nvidia-smi
```

在有桌面的Linux下安装比较麻烦，需要先关闭桌面再安装，如果这是你的主力机的话，建议用docker。

### 安装CUDA

使用PyTorch之前，需要为其配置好CUDA环境。

前往[官网](https://developer.nvidia.com/cuda-toolkit-archive)下载，建议下载[10.2](https://developer.nvidia.com/cuda-10.2-download-archive)版本，因为[PyTorch](https://pytorch.org/get-started/previous-versions/)基本上每个版本都会对10.2进行适配。

选择对应系统和local版本，离线安装比较省事。选择好后底下会有安装步骤，按照官网提示操作即可。下载较慢，时常丢包，需耐心等待或者科学上网。装好后验证：

```shell
nvcc -V
```

很罕见的会报错，则需要进行环境变量配置，上网查一查吧。Windows就是点点点下一步；Linux下安装一路都会有提示，照着他走就可以。

有的教程说还需要安装cudnn，实测不用，只要conda或者pip安装**cudatoolkit**就行了。

安装CUDA时，我的Linux系统报过两个错，运行了以下命令后方可，权做记录：

```shell
pip uninstall nvidia_cublas_cu11
apt-get install nvidia-modprobe
```

## 安装PyTorch

[PyTorch](https://pytorch.org/)官网主页提供了安装的命令，但更多时候需要根据安装的CUDA来决定版本。前往[历史版本](https://pytorch.org/get-started/previous-versions/)，<kbd>ctrl+f</kbd>搜索10.2，找到CUDA10.2的命令行，如：

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

Windows系统在<kbd>Anaconda prompt</kbd>中安装，linux在命令行终端安装即可。

最好从官网安装，因为曾经的各个镜像只有CPU版本。不知道现在有没有加入GPU版本的，我没有试过。命令行输入python回车，依次输入：

```python
import torch
torch.cuda.is_available()
```

输出True则安装成功。

本文的目的是说明开展深度学习需要做哪些准备，不涉及特别细节的内容，如果您对安装有顾虑，完全可以搜每节的标题，去对照那些有图的详细教程安装喔～
