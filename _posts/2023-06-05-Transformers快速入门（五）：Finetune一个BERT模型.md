---
title: Transformers快速入门（五）：Finetune一个BERT模型

date: 2023-06-05 17:00:00 +0800

categories: [Deep Learning, Transformers]

tags: [deep learning, transformers]

math: true

mermaid: true

description: Transformers入门，Huggingface，Finetune，BERT
---

一切前期工作准备就绪，本章我们来Finetune一个自己的BERT模型。

Finetune翻译过来是微调，指把别人训练好的现成的模型拿过来，用自己的数据，调整一下参数，再继续进行训练，使之符合自己数据的特点。微调是一种迁移学习（Transfer Learning），由于前人花很大精力训练出来的模型，在大概率上会比自己从零开始搭的模型要强悍，所以很多时候，我们都在既有的模型上继续调整，也就是迁移了原模型的能力。

虽然Transformers包为我们提供了极其方便的训练方法[Trainer](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.Trainer)，官方也提供了基于Trainer的[微调](https://huggingface.co/docs/transformers/v4.30.0/en/tasks/sequence_classification)方法，但是越简单步骤越少的，往往越不自由，不利于我们学习和魔改。所以，我们采用折中的办法，在尽可能详细展示训练细节的情况下，做到代码的简洁。

## 任务目标

Finetune BERT，进行文本的情感分类。

由于BERT本身并没有分类层，因此直接使用BERT分类是完全随机的。Finetune是在训练这分类层。

## 项目结构

依托PyTorch开发的深度学习项目，基本都符合这样的文件结构，即：

- data/dataset（数据文件夹）
- ckpts（预训练模型文件夹）
- data/processor.py（数据读取、预处理，有时会写进utils.py）
- model.py（定义模型结构）
- train/run/main.py（模型训练、验证）
- test/inference.py（模型预测/推理，有时会写进run.py）
- options/config/args.py（参数）
- utils.py（其他工具，如日志输出）

```shell
BERT_finetune
├── data
├── ckpts
├── processor.py
├── model.py
├── run.py
├── config.py
├── utils.py
├── requirements.txt
```

## processor.py——数据定义和读取

首先定义数据，以json文件为例，每一行格式如下：

```
{"news": "“双节”来临，各大金店逐步进入到一年当中的消费高峰。年轻家庭成为黄金消费的主力，2022年的第一天，在北京一家老牌金店开门前，门口就已经有不少顾客排队等候入场。随着2021年下半年，黄金价格波动整体趋缓，消费者的投资热情显著高于疫情前水平。菜百股份高级黄金投资分析师李洋：这张图是2021年全年的国际金价，年初开盘1900美元左右，随着一季度震荡下跌，二季度一个V型的反转，下半年金价围绕1800美元区间上下波动，全年收盘在1829美元。金价每次在下跌过程中，投资者都会集中购买投资产品，作为家庭资产的储备。", "label": 1}
```

"label"的1代表正向情感，0代表负。其它格式同理，只需保证一条文本对应一个情感即可。

前文我们已经学习了PyTorch的数据处理，这里就不赘述了，直接上代码：

```python
import json
from torch.utils.data import Dataset, DataLoader

class FinData(Dataset):
    def __init__(self, file_name):
        data = [json.loads(line) for line in open(file_name, "r").readlines()]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        news, label = self.data[idx].values()
        return {
            'news': news,
            'label': label,
        }

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    def collote_fn(batch_samples):
        batch_sentence, batch_label = [], []
        for sample in batch_samples:
            batch_sentence.append(sample['news'])
            batch_label.append(int(sample['label']))
        batch_inputs = tokenizer(
            batch_sentence,
            max_length=args.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return {
            'batch_inputs': batch_inputs,
            'labels': batch_label
        }

    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle,
                      collate_fn=collote_fn)
```

## model.py——模型结构定义

BERT用于分类的通常做法，是用BERT的`[CLS]`token即以下代码的`last_hidden_state[:, 0, :]`，作为分类向量，后接一个分类器。

```python
from torch import nn
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel

class BertForCLS(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        outputs = self.bert(**batch_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits
```

以上代码清晰定义了分类器。当然，也可以直接使用：

```python
from transformers import BertForSequenceClassification
```

只不过这样的话，这个黑盒就更加黑了～

## config.py——参数

argparse使得我们可以通过命令行传入参数。

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.",
                        )
    parser.add_argument("--train_file", default="./data/train_set.jsonl", type=str, help="The input training file.")
    parser.add_argument("--dev_file", default="./data/dev_set.jsonl", type=str, help="The input evaluation file.")
    parser.add_argument("--test_file", default="./data/test_set.jsonl", type=str, help="The input testing file.")
    
    parser.add_argument("--device", default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_checkpoint",
                        default="./bert-base-chinese/", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models",
                        )
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to save predicted labels.")

    # Training parameters
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=8, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training."
                        )
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some."
                        )
    args = parser.parse_args()
    return args
```

## utils.py——一些有用的小工具

我通常习惯在这里定义随机种子和日志工具。日志使得模型训练过程可视化，我写的这个日志工具非常优雅，治好了我多年的强迫症。

```python
import logging
import torch
import random
import numpy as np
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def set_logger(args):
    # 格式化日志
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    # 打开指定的文件并将其用作日志记录流
    file_handler = logging.FileHandler(args.output_dir + "logs.log")
    file_handler.setFormatter(formatter)
    # 记录控制台的输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    # 记录日志
    logger = logging.getLogger(args.model_type)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger
```

## run.py——训练、验证、预测

```python
from processor import FinData, get_dataLoader
from model import BertForCLS
from utils import seed_everything, set_logger
from config import parse_args

import os
import json
import time
import torch
from sklearn.metrics import classification_report
from transformers import get_scheduler
from transformers import AutoConfig, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from tqdm.auto import tqdm

MODEL_CLASSES = {
    'BertForCLS': BertForCLS,
}

# 将模型放入设备
def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data

# 每个batch放入模型训练
def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    finish_step_num = epoch * len(dataloader)
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'Epoch {epoch} loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

# 验证、测试每个batch
def test_loop(args, dataloader, model, mode='test'):
    assert mode in ['dev', 'test']
    all_pred = []
    all_label = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]

            all_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_label.extend(batch_data['labels'].cpu().numpy().tolist())

    return classification_report(all_label, all_pred, output_dict=True, zero_division=1)

# 训练每个epoch
def train(args, train_dataset, dev_dataset, model, tokenizer):
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    best_score = 0.
    best_model = None
    save_weight = None
    for epoch in range(args.num_train_epochs):
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, model, mode='dev')
        # macro_f1, micro_f1 = float(metrics['macro avg']['f1-score']), float(metrics['weighted avg']['f1-score'])
        # dev_f1_score = (macro_f1 + micro_f1) / 2
        acc = metrics['accuracy']
        print("\n")
        logger.info(f"Dev Accuracy: {(100 * acc):>0.4f}%")
        if acc > best_score:
            best_score = acc
            save_weight = f'epoch_{epoch + 1}_dev_acc_{(100 * best_score):0.1f}_weights.bin'
            best_model = model.state_dict()
    logger.info(f'saving new weights to {args.output_dir}...\n')
    torch.save(best_model, os.path.join(args.output_dir, save_weight))
    logger.info("Training done!")

def test(args, test_dataset, model, tokenizer, save_weights):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        time.sleep(0.1)
        logger.info(f"Test Accuracy: {(100 * metrics['accuracy']):>0.4f}%")
        for avg in ["macro avg", "weighted avg"]:
            for metric in ["f1-score", "precision", "recall"]:
                logger.info(f"Test {avg} {metric}: {(100 * metrics[avg][metric]):>0.4f}%")

def predict(args, sent, model, tokenizer):
    inputs = tokenizer(
        sent,
        max_length=args.max_seq_length,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    pred = int(logits.argmax(dim=-1)[0].cpu().numpy())
    prob = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    return pred, prob[pred]

def del_file(path_data):
    for i in os.listdir(path_data):
        file_data = path_data + i
        if os.path.isfile(file_data):
            os.remove(file_data)
        else:
            del_file(file_data)

if __name__ == '__main__':
    args = parse_args()
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        del_file(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    logger = set_logger(args)
    seed_everything(args.seed)
    logger.info(f'Loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    args.num_labels = 2
    model = MODEL_CLASSES[args.model_type].from_pretrained(
        args.model_checkpoint,
        config=config,
        args=args
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = FinData(args.train_file)
        dev_dataset = FinData(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = FinData(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = FinData(args.test_file)
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
            logger.info(f'predicting labels of {save_weight}...')

            results = []
            model.eval()
            for s_idx in tqdm(range(len(test_dataset))):
                sample = test_dataset[s_idx]
                pred, prob = predict(args, sample['news'], model, tokenizer)
                results.append({
                    "news": sample['news'],
                    "label": sample['label'],
                    "pred_label": str(pred),
                    "pred_prob": prob
                })
            with open(os.path.join(args.output_dir, save_weight + '_test_pred.json'), 'wt', encoding='utf-8') as f:
                for sample_result in results:
                    f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
```

以上代码有一些未详尽的解释，有：

1. AdamW优化算法，包括其参数weight_decay、lr、betas、eps等；
2. scheduler，包括其参数warmup。

希望读者先自行查询学习，有机会再填这个坑～

## 运行

最后，在命令行中进入该目录，运行：

```shell
python run.py \
    --output_dir=./Bert_results/ \
    --model_type=BertForCLS \
    --model_checkpoint=./ckpts/bert-base-chinese \
    --train_file=./data/train_set.jsonl \
    --dev_file=./data/dev_set.jsonl \
    --test_file=./data/test_set.jsonl \
    --max_seq_length=512 \
    --batch_size=8 \
    --do_train \
    --do_test
```

## 小结

本节我们介绍了Finetune一个BERT模型的全流程，代码和数据将会发布在[GitHub](https://github.com/JinHanLei/Transformers_tutorial)。

距离BERT论文发表已经过去5年，或许其性能早已被后来者超越，但其中蕴含的思想仍值得我们学习和继续探索。本文虽然就训练流程和模型结构进行了较为详细的呈现，但也是直接使用了`BertPreTrainedModel`作为基底模型，仍然是一个黑盒，建议同学们就BERT模型本身继续深入探索，彻底掌握其精髓。

自此，这个系列更新结束。
