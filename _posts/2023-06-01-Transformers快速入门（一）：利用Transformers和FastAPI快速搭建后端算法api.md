---
title: Transformers快速入门（一）：利用Transformers和FastAPI快速搭建后端算法api

date: 2023-06-01 17:00:00 +0800

categories: [Deep Learning, Transformers]

tags: [deep learning, transformers]

comments: false

description: Transformers入门，Huggingface，pipelines，FastAPI，后端算法api
---

如果你对自然语言处理 (NLP, Natural Language Processing) 感兴趣，想要了解ChatGPT是怎么来的，想要搭建自己的聊天机器人，想要做分类、翻译、摘要等各种NLP任务，[Huggingface](https://huggingface.co)的[Transformers](https://huggingface.co/docs/transformers)库是你不能错过的选择。在Transformers的帮助下，许多任务只需几行代码便能轻松解决。本章将带你体验Transformers的魅力。

## 预训练模型

在Transformer结构中，存在许多的$$W$$权重矩阵，训练后具有大量的信息，并且可以保存为文件。如果有大佬训练了效果特别好的模型，他能不能慷慨分享出来，给我们用？幸运的是，真的存在这样一群具有无私**开源精神**的人，为推动科技发展造福全人类而不懈努力，这便是Huggingface社区的贡献者们。

让我们进入Huggingface的模型库[>>>点击直达](https://huggingface.co/models)一探究竟。

印入眼帘的便是大量的预训练模型，包括著名的[BERT](https://huggingface.co/bert-base-uncased)、[GPT2](https://huggingface.co/gpt2)等，你常可以看到以下字眼：

- large/base/small/tiny/mini：指模型大小
- cased/uncased：cased指区分大小写，"Hi"和"hi"是两个不同的词；uncased指预先对文本做lower-case，"Hi"和"hi"都会转成"hi"，于是词表只有小写的单词
- zh/fr/en...：指语言，中文/法语/英语等
- wwm：Whole Word Masking，对全词进行Mask

点进[GPT2](https://huggingface.co/gpt2)看看。

在模型首页有三个选项卡<kbd>Model Card</kbd>|<kbd>Files</kbd>|<kbd>Community</kbd>，作用如下：

- Model Card：介绍这个模型，对应Files里的README.md。一般都会提供`How to use`，能用Transformers的几行代码把此模型跑起来。有些模型的右侧会提供`Hosted inference API`，直接在线尝试使用这个模型；底下还有`Spaces`，指使用了这些模型的空间，设计了界面，不过俩蛮多时候都不好使
- Files：保存了模型的各个文件。点击中间的下载图标即可下载，我试了右键另存为不太行。用PyTorch必要下载：
  1. pytorch_model.bin，模型的参数文件，最大（其他大文件基本都是针对其他类深度学习框架，如TensorFlow的tf_model.h5，不用下载）
  2. config.json，保存模型结构、参数等信息
  3. vocab.txt(.json)，词表文件
  4. 其他小文件也一并下载
- Community：提供模型的讨论区，结合了GitHub的issue和PR，对模型的问题或改进可以提在这

### Git LFS

我在浏览器下载模型文件经常会断掉，只能再从头下载，很令人抓狂，好在Huggingface提供了命令行下载的方式：`git lfs`。

模型文件很大，但Git无法很好地下载大文件，于是诞生了Git Large File Storage (LFS)这个工具，安装也很简单，只需到[官网](https://git-lfs.com/)下载安装文件，按照GitHub提供的[教程](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux)安装就行，基本上就是一直下一步。下载模型用`git clone`加上模型网站地址即可，如：

```shell
git clone https://huggingface.co/gpt2
```

下载[数据集](https://huggingface.co/datasets)同理。

## 开箱即用的pipeline

在Model Card中常见：

```python
from transformers import pipeline
```

使用pipeline，首先需要安装transformers，在pytorch环境配置好的情况下，只需：

```shell
pip install transformers
```

[pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)封装了预训练模型和对应的前处理和后处理环节，可以直接进行包括但不限于以下任务：

| **任务**                     | **描述**                                       | **模态**        | **Pipeline使用方式**                          |
| ---------------------------- | ---------------------------------------------- | --------------- | --------------------------------------------- |
| Text classification          | 给一段文本赋上标签                             | NLP             | pipeline(task=“sentiment-analysis”)           |
| Text generation              | 生成文本                                       | NLP             | pipeline(task=“text-generation”)              |
| Summarization                | 生成一段文本或文档的摘要                       | NLP             | pipeline(task=“summarization”)                |
| Image classification         | 给图片赋上标签                                 | Computer vision | pipeline(task=“image-classification”)         |
| Image segmentation           | 给图片的每个像素赋上标签                       | Computer vision | pipeline(task=“image-segmentation”)           |
| Object detection             | 预测图像中的对象边界和类别                     | Computer vision | pipeline(task=“object-detection”)             |
| Audio classification         | 给音频赋上标签                                 | Audio           | pipeline(task=“audio-classification”)         |
| Automatic speech recognition | 语音转文本                                     | Audio           | pipeline(task=“automatic-speech-recognition”) |
| Visual question answering    | 给定图片和问题，给出回答                       | Multimodal      | pipeline(task=“vqa”)                          |
| Document question answering  | 给定带有文档（例如表格）的图片和问题，给出回答 | Multimodal      | pipeline(task=“document-question-answering”)  |
| Image captioning             | 给图片起个标题                                 | Multimodal      | pipeline(task=“image-to-text”)                |

### Pipeline使用案例

例如情感分析，我们只需要输入文本，就可以得到其情感标签（积极/消极）以及对应的概率：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
```

pipeline会自动完成以下三个步骤：

1. 将文本预处理为模型的输入格式
2. 将预处理好的文本送入模型
3. 对模型的预测值进行后处理，输出想要的结果

pipeline会自动选择合适的预训练模型来完成任务。例如对于情感分析，默认就会选择微调好的英文情感模型 *distilbert-base-uncased-finetuned-sst-2-english*。

要想指定模型，可以加入model参数，如（但我试了，会报网络错误）：

```python
generator = pipeline("text-generation", model="distilgpt2")
```

Transformers 库在首次加载时某模型时，会自动缓存模型到`~/.cache/huggingface`，再次使用时，直接调用缓存好的模型。

如果你也不喜欢这种被人瞒着的感觉，最好下载模型，在本地调用。比如下载好了`bert-base-chinese`到当前目录，再运行：

```python
classifier = pipeline('text-classification', model='./bert-base-chinese')
```

这时会报一个warning，说权重没有初始化什么的，这是因为bert本身没有对文本分类任务做训练，所以没有关于分类的权重，需要自己训练，这个会在后面的章节进行学习。

## FastAPI搭建后端算法接口

FastAPI是最快的基于 Python 的 Web 框架之一。相比于Flask，有以下特点：

- FastAPI使用 Pydantic 进行数据验证
- Uvicorn提供异步请求能力

先安装：

```shell
pip install fastapi uvicorn
```

以翻译任务为例，首先新建一个`api.py`文件，用transformers的pipeline搭建模型：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

TRANS_MODEL_PATH = 'liam168/trans-opus-mt-en-zh'
def trans(text):
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL_PATH)
    translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
    return translation(text, max_length=400)
```

用pydantic定义数据格式，翻译任务就只需要接收**模型名称**和**输入文本**：

```python
from pydantic import BaseModel

class Data(BaseModel):
    model_name: str
    input: str
```

FastAPI构建服务：

```python
from fastapi import FastAPI

app = FastAPI()
@app.post('/api/trans')
def translate(data_zh: Data):
    res = {
        "code": 200,
        "message": "OK"
    }
    msg = data_zh.input
    if data_zh.model_name == 'opus':
        trans_res = trans(msg)
        res.update({data_zh.model_name: trans_res[0]['translation_text']})
    return res
```

Uvicorn跑起服务：

```python
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app="api:app", host="127.0.0.1", port=8000, reload=True)
```

其中api指这份代码的文件名`api.py`；host指ip"127.0.0.1"是本地的地址；port端口8000；reload是调试模式，可以在服务跑着的情况下改代码热重载，不用每次改完代码都关掉再重新跑了。

测试服务：

```python
import requests

data_bin = {"model_name": "opus", "input": "I like to study Data Science and Machine Learning."}
res = requests.post("http://127.0.0.1:8000/api/trans", json=data_bin).json()
print(res)
```

不出意外的话就能输出结果了。

## 小结

在本章中，我们初步了解了如何使用Transformers提供的pipeline来处理NLP任务，并且用FastAPI搭建了一个简单的服务。

在下一章中，我们会具体介绍Models和Tokenizers的参数以及使用方式。
