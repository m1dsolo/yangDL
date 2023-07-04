# yangDL

轻量级Pytorch封装框架，封装了深度学习中重复的代码，可以简化代码逻辑，从而实现快速地开发。

Features:

1. 简化代码逻辑，减少代码量。
2. 多折train, val, test，自动汇聚多折结果平均值。
3. 自动将结果写入TensorBoard中。
4. EarlyStop。
5. 可定制化进度条，终端实时显示运行结果。
6. 简洁的代码可用于入门深度学习。
7. 模型封装(比如SAM)

Metrics:

1. 二分类：
    - 支持指标：[acc, pos\_acc, neg\_acc, precision, recall, sensitivity, specificity, f1\_score, auc, ap, thresh]
    - 支持选择阈值使得f1\_score或者roc\_auc最大。

2. 多分类：
    - 支持指标：[acc, precision, recall, sensitivity, specificity, f1\_score, auc]

3. 二类分割：
    - 支持指标[acc, pos\_acc, neg\_acc, precision, recall, sensitivity, specificity, f1\_score, dice, iou, thresh]

4. 多类分割：
    - 支持指标[acc, precision, recall, sensitivity, specificity, f1\_score, dice, iou]

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Todo](#todo)
- [License](#license)

## Background

研一时写了许多深度学习相关的代码，发现代码的很多流程是相似的，就萌生了想把这些代码抽象出一个框架的想法。
后面也逐渐了解到[Pytorch Lightning](https://github.com/Lightning-AI/lightning)等库已经做了这件事，故抱着学习Python语法、Pytorch框架以及代码设计等心态来实现了这个轻量级Pytorch封装框架。

## Install

```bash
pip install yangDL
```

## Usage

参考[代码](./example/digit-recognizer.py)，是[kaggle数字识别竞赛](https://www.kaggle.com/competitions/digit-recognizer)的一个实现。

## Todo

1. 实现DDP，支持多机多卡训练。
2. save ckpt时save optimizer和lr_scheduler(为了恢复checkpoint继续训练)
3. 实现检测和实例分割Metrics(比如mAP)

## License

[MIT](LICENSE) © m1dsolo
