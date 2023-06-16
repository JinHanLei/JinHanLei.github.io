---
title: Transformers快速入门（三）：用PyTorch加载和处理数据

date: 2023-06-03 17:00:00 +0800

categories: [Deep Learning, Transformers]

tags: [deep learning, transformers]

comments: false

math: true

mermaid: true

description: Transformers入门，Huggingface，PyTorch，Dataset，DataLoader
---

Transformers库建立在[PyTorch](https://pytorch.org/)框架之上，需要通过PyTorch来加载和处理数据、定义优化器、定义和调整模型参数，甚至直接加载模型等等。本章将介绍用PyTorch加载和处理数据。

我一直都说，人类相比于动物的一大区别就是人类善于使用工具，所谓君子生非异也，善假于物也。而人类创造出各种工具，是为了方便我们，而不是折磨我们。会有人说PyTorch好难，但是不用PyTorch，深度学习会更难！所以，让我们快速进入PyTorch这一工具的学习吧！

## PyTorch张量运算和自动微分

[PyTorch](https://pytorch.org/)由Facebook人工智能研究院于2017年推出，具有强大的GPU加速张量计算功能，并且能够自动进行微分计算，从而可以使用基于梯度的方法对模型参数进行优化，大部分研究人员、公司机构、数据比赛都使用PyTorch。

### 张量创建

在深度学习领域你会经常看到张量（Tensor）的表述，张量是深度学习的基础，所以谷歌会把他的深度学习框架叫做[TensorFlow](https://www.tensorflow.org/)。深度学习中的张量可以理解成数组，类似[numpy](https://numpy.org/)的array。例如：

- 单个数字就是0维张量，称为标量（scalar）；
- 1维张量称为向量（vector）；
- 2 维张量称为矩阵（matrix）；
- 再多点维度就统一称作张量了。

高等代数中学习过矩阵运算，就是最基本的张量运算。

在用Transformers时最常见的是二维和三维张量。二维张量一般是权重矩阵$W$等，三维张量一般是原数据处理成$batchsize \times 序列长度 \times 模型维度$。在描述张量维度时，或者创建多维张量时，你会经常看到$W\in\mathbb{R}^{d_m \times d_k \times d_h}$这种类似表述，用*几行几列*这样的方式去理解的话，相当不直观。

分享一个自创的**框框理论**，$d_m \times d_k \times d_h$代表最大一个框包着$m$个框、再下一层有$k$个，最里层有$h$个。第零维$m$个框：

$$
\begin{array}{c}
m个 \\
[\overbrace{[...],[...],...,[...]}]
\end{array}
$$

第一维$k$个框：

$$
\begin{array}{}
k个 \\
[[\overbrace{[...],...,[...]}],...,]
\end{array}
$$

第二维$h$个框：

$$
\begin{array}{}
h个 \\
[[[\overbrace{[...],...,[...]}],...],...]
\end{array}
$$

值得注意的是，这个维度并不是$1 \times d_m \times d_k \times d_h$喔，因为最外面必需有个大框包起来，不然不漏了吗～

PyTorch提供了多种方式来创建张量，以创建一个$2 \times 3$的矩阵为例：

```python
import torch
# empty作用就是初始化一块内存放着，里面数据不重要，根本不会用
t = torch.empty(2, 3)
# 随机初始化张量，范围是[0,1)
t = torch.rand(2, 3)
# 随机初始化张量，服从标准正态分布
t = torch.randn(2, 3)
# 全0矩阵，其中的0是长整型，也可以换成torch.double、torch.float64
t = torch.zeros(2, 3, dtype=torch.long)
# 同理有全1矩阵
t = torch.ones(2, 3, dtype=torch.long)
```

上面比较常用的是全0和全1，对判断真假很有用。也可以从一个张量创造维度相同的张量，`like`一下：

```python
import torch
t = torch.empty(2, 3)
x = torch.rand_like(a)
x = torch.randn_like(a)
x = torch.zeros_like(a)
x = torch.ones_like(a)
```

也可以通过基于已有的数组创建张量：

```python
# 从列表
_list = [[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]]
t = torch.tensor(_list)
# 从ndarray
import numpy as np
array = np.array([[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]])
t = torch.from_numpy(array)
```

这样创建的张量默认在*CPU*，将其调入*GPU*有如下方式：

```python
t = torch.empty(2, 3).cuda()
t = torch.empty(2, 3, device="cuda")
t = torch.empty(2, 3).to("cuda")
```

默认是使用当前第0张卡，指定用第1张卡：

```python
t = torch.empty(2, 3).cuda(1)
t = torch.empty(2, 3, device="cuda:1")
t = torch.empty(2, 3).to("cuda:1")
```

对应的可以调入*CPU*：

```python
t = torch.empty(2, 3).cpu()
t = torch.empty(2, 3, device="cpu")
t = torch.empty(2, 3).to("cpu")
```

### 张量运算

张量的加减乘除、拆拼换调、特殊函数，都能在PyTorch找到快速方法。

#### 加减乘除

```python
x = torch.rand(2, 3)
y = torch.rand(2, 3)
# 等价于x + y
z = torch.add(x, y)
# torch没有减方法，但是可以x - y
# 矩阵点乘，multiplication，Hadamard积，等价于x * y
z = torch.mul(x, y)
# 矩阵叉乘，矩阵乘法，matrix multiplication，等价于x @ y
z = torch.mm(x, y)
# 会报错，因为两者的维度不能做叉乘，需要如下转置
z = torch.mm(x, y.T)
# 三维对应矩阵乘法，batch matrix multiplication
x = torch.rand(2, 3, 4)
y = torch.rand(2, 4, 3)
z = torch.bmm(x, y)
# 更普遍的矩阵叉乘
z = torch.matmul(x, y)
# 除法不常用，但也可以x / y
```

#### 广播机制

前面我们都是假设参与运算的两个张量形状相同，但是PyTorch同样可以处理不相同形状的张量。

```python
x = torch.ones(2, 3, 4)
y = torch.ones(1, 3, 4)
z = x + y
```

PyTorch会使得最外面的框维度相同，做法是复制，如上例的$y$复制一份变成$2 \times 3 \times 4$，然后以此类推使得前面的框框都相同，最后可以做相同维度运算。再来个更极端的例子：

```python
import torch
x = torch.ones(2, 1, 3, 4)
y = torch.ones(5, 4, 3)
z = torch.matmul(x, y)
print(z)
```

这么乱都能乘？耶斯。

1. 首先来看，不乱的是最后两位的$3 \times 4和4 \times 3$，刚好能做叉乘，好，所以结果的最后两位是$3 \times 3$。
2. 再看前面的维度，$y$少了框，先补最外面$y$变成$2 \times 5 \times 4 \times 3$。
3. 这时$x$第二维的$1$少了，复制成$2 \times 5 \times 3 \times 4$，这样就可以乘了。

聪明的你要问，如果$x$第二维是$3$，复制不成$5$啊，那怎么办？怎么办？难办就别办了！答案就是会报错。

#### 拆拼换调

这些方法几乎是最常用的，跟着我好好理解一遍哦。首先是**拼接**的`cat`方法：

```python
x = torch.tensor([[1, 2, 3], [ 4,  5,  6]], dtype=torch.double)
y = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.double)
z = torch.cat((x, y), dim=0)
```

看到`dim=0`了吗，根据**框框理论**，这是把第0维的几个框框拼起来，得到：

```python
tensor([[ 1.,  2.,  3.],
     [ 4.,  5.,  6.],
     [ 7.,  8.,  9.],
     [10., 11., 12.]], dtype=torch.float64)
```

当`dim=1`，则是把第一个框框里的拼起来，得到：

```python
tensor([[ 1.,  2.,  3.,  7.,  8.,  9.],
     [ 4.,  5.,  6., 10., 11., 12.]], dtype=torch.float64)
```

**拆分**就用索引与切片，操作如同`list`。

```python
# 取第0维第1个框里的第2位，注意第X是从0开始
t = torch.randn(3, 4)
x = t[1, 2]
# 取第0维所有框里的第2位
x = t[:, 2]
# 取第0维所有框里的第2、3、4位赋值为100
t[:, 2:4] = 100
```

**转换**有多种操作，如下：

- `view`将张量转换为新的形状，需要保证总的元素个数不变。

  ```python
  # x.shape为6
  t = torch.arange(6)
  # 2×3
  x = t.view(2, 3)
  # -1会自动计算，如下例是3×2
  x = t.view(-1, 2)
  ```

- `transpose`交换张量中的两个维度，参数为相应的维度。

  ```python
  # 2×3的张量
  t = torch.tensor([[1, 2, 3], [4, 5, 6]])
  # 调换成3×2
  x = t.transpose(0, 1)
  # 这就是矩阵转置，同x.t()或者x.T
  ```

- `permute`可以直接设置新的维度排列方式，而`transpose`每次只能交换两个维度。

  ```python
  # 1×2×3的张量
  t = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
  # 换成3×1×2
  x = t.permute(2, 0, 1)
  ```

- `reshape`，与`view`功能几乎一致。区别在于进行`view`的张量必须是连续的，可以调用`.is_contiguous()`来判断张量是否连续；如果非连续，需要先通过`.contiguous()`函数将其变为连续的，再`view`。但`reshape`一步到位。

  ```python
  t = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
  t = t.permute(2, 0, 1)
  # 这样会报错，因为x.shape是3×1×2，不连续
  x = t.view(-1)
  # contiguous
  x = t.contiguous()
  x = x.view(-1)
  # 直接用reshape
  x = t.reshape(-1)
  ```

**降维与升维**：很多时候，深度学习模型无法输入向量、矩阵，只能输入张量，所以要用`.unsqueeze()`把一二维的升维成三维以上张量，反之`.squeeze()`降维。*squeeze*就是挤压、压榨的意思，让张量变弱了，框框都压没了，很形象吧！

```python
t = torch.tensor([1, 2, 3, 4])
# 最外面套个框框
x = torch.unsqueeze(t, dim=0)
# 或者
x = t.unsqueeze(dim=0)
# 思考下dim=1是什么样子？
# squeeze会挤掉所有为1的维度，比如3×1×2就会变成3×2
x = t.squeeze()
```

### 特殊函数

PyTorch提供了许多内置函数，只需要点一下，方法就可以出来，例如常用的：

- 包括均值`.mean()`、方差`.std()`、平方根`.sqrt()`、对数`.log()`等等常见的数学运算，以平均值为例。

  ```python
  t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
  # 默认情况下计算所有元素的平均值
  m = t.mean()
  # 拿掉最外的框框，里面第0维的框框对应位置平均，这里就是对列平均
  m = t.mean(dim=0)
  # dim=1呢？
  # torch.mean()是相同效果
  m = torch.mean(t)
  ```

- 三角函数`.sin().cos().tan()`等。

- 激活函数，一般不在这里直接用，而是用`torch.nn`里的激活函数，但也放出来吧。

  ```python
  a = torch.sigmoid(t)
  a = torch.tanh(t)
  ```

- 对角线。

  ```python
  # 如果输入是一个向量，返回这个向量为对角线元素的2D方阵；如果是方阵，返回其对角线的1D张量
  t = torch.randn(3)
  d = torch.diag(t)
  # 如果不是方阵呢？试试看
  # 返回方阵的迹
  t = torch.arange(1, 10).view(3, 3)
  d = torch.trace(t)
  # 返回矩阵下三角，其他置0
  d = torch.tril(t)
  # 创建对角线为1,其他为0的方阵
  t = torch.eye(3)
  ```

### 自动微分

Pytorch提供自动计算梯度的功能，只需要执行`.backward()`。

```python
x = torch.tensor([2.], requires_grad=True)
y = torch.tensor([3.], requires_grad=True)
z = (x + y) * (y - 2)
z.backward()
print(x.grad, y.grad)
```

很容易手工求解$\frac{\text{d}z}{\text{d}x} = y-2,\frac{\text{d}z}{\text{d}y} = x + 2y - 2$，当$x=2,y=3$时，$\frac{\text{d}z}{\text{d}x}=1$和$\frac{\text{d}z}{\text{d}y}=6$，与代码计算结果一致。

## PyTorch加载和处理数据

讲到这，相信你已经大致掌握了PyTorch怎么创建和运算张量，让我们马上进入模型前的最后一步——数据的加载和处理。

Pytorch提供了`Dataset`/`IterableDataset`，和`DataLoader`和用于处理数据，它们既可以加载Pytorch预置的数据集，也可以加载自定义数据。其中`Dataset`/`IterableDataset`负责存储样本以及它们对应的标签；数据加载类`DataLoader`负责迭代地访问数据集中的样本。

为什么用这些方法，而不直接创建Tensor喂给模型？个人使用中有这些感受：

1. 符合End2End的理念，代码可以干净整洁、易于理解和上手；
2. 能规范储存样本和标签，读取和处理时也很方便；
3. 能快速调用其内置的方法，如分`batch`、打乱等等；
4. 对于数据量超大的情况，能够以迭代器的方式处理。

下面进行这些组件的介绍。

### Dataset

`Dataset`类的本质是`索引-样本`，这样我们就可以方便地通过`dataset[idx]`来访问指定索引的样本。

`Dataset`必须重写`__getitem__()`函数来指定获取样本的方式，因为源码中这个方法指定了`NotImplementedError`，不实现就报错。一般还会实现`__len__()`用于返回数据集的大小。例如将一个存有股票的`csv`文件转为`Dataset`：

```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code, date, open, close, high, low, is_rise= self.df.iloc[idx, :].values
        return {
            "stock": [open, close, high, low],
            "label": is_rise,
        }

ds = MyDataset("stock.csv")
```

### IterableDataset

当数据量超级大，或者访问远程服务器产生的数据，不可能一把梭到内存里，所以用`IterableDataset`迭代地访问数据集。必须重写`__iter__()`函数，用于返回一个样本迭代器。

```python
class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
print(list(DataLoader(ds, num_workers=0)))
```

如果要配合`DistributedDataParallel`进行多进程分布式训练，`num_workers=0`就可以了。如果非要多个`workers`，例如`num_workers=2`，会出问题，由于每个`workers`都获取到了单独的数据集拷贝，因此会重复访问每一个样本。需要定义每一个`workers`应该获取哪些数据：

```python
from torch.utils.data import get_worker_info

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    # 把数据复制给worker_info
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    # 根据workers数分割数据
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
```

### DataLoaders

在加载数据后，需要将数据集切分为很多batches，然后按batch将样本喂给模型，并且循环这一过程，每一个完整遍历所有样本的循环称为一个*epoch*。`DataLoader`类专门负责处理这些操作。紧接着`IterableDataset`的例子：

```python
train_dataloader = DataLoader(ds, batch_size=2)
for i in train_dataloader:
    print(i)
```

DataLoader中还有几个值得探讨的参数，`sampler`和`collate_fn`。

#### sampler

`sampler`用于控制数据的顺序。对于`IterableDataset`，数据本身是一个个按顺序来的，所以无法使用，会报错。对于`Dataset`，可以调整，例如：

```python
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

class MyDataset(Dataset):
    def __init__(self):
        self.examples = list(range(10))
        self.labels = list(range(10))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "example": self.examples[idx],
            "label": self.labels[idx],
        }

ds = MyDataset()
train_sampler = RandomSampler(ds)
test_sampler = SequentialSampler(ds)

train_dataloader = DataLoader(ds, batch_size=2, sampler=train_sampler)
print(list(train_dataloader))
test_dataloader = DataLoader(ds, batch_size=2, sampler=test_sampler)
print(list(test_dataloader))
```

一般在训练时需要打乱数据，而测试时不用。上例输出`train_dataloader`已经被打乱了，而`test_dataloader`则没有。当然，也可以直接在`DataLoader`中指定参数`shuffle=True`来打乱，但总归这样自由一些。

#### collate_fn

批处理函数`collate_fn`负责对每一个采样出的 batch 中的样本进行处理。默认的 `collate_fn` 会进行如下操作：

- 添加一个新维度作为batch维；
- 自动将NumPy数组和Python数值转换为PyTorch张量；
- 保留原始的数据结构，例如输入是字典的话，它会输出batch后的字典。

例如上例的`test_dataloader`：

- `{'example': 0, 'label': 0}, {'example': 1, 'label': 1}`

变成了2个batch和tensor格式的：

- `{'example': tensor([0, 1]), 'label': tensor([0, 1])}`

我们也可以传入手工编写的`collate_fn`函数以对数据进行自定义处理。例如给每个样本乘以10：

```python
def collote_fn(batch_samples):
    batch_example, batch_label = [], []
    for sample in batch_samples:
        batch_example.append(sample["example"] * 10)
        batch_label.append(sample["label"])
    return {
        "batch_inputs": batch_example,
        "labels": batch_label
    }

ds = MyDataset()
train_dataloader = DataLoader(ds, batch_size=2, collate_fn=collote_fn)
print(list(train_dataloader))
```

`collote_fn`输入的是每个batch。遍历batch里的字典，处理成你想要的就可以了。

## 小结

本章我们熟悉了PyTorch的数据处理，终于搞定数据了！下一章我们进入模型的加载和修改。
