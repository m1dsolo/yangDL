# yangDL

轻量级Pytorch封装框架，封装了深度学习中重复的代码，可以简化代码逻辑，从而实现快速地开发。

Features:

1. 多折train, val, test，自动汇聚多折结果平均值。
2. 自动将结果写入TensorBoard中。
3. EarlyStop。
4. 二分类时自动选择最佳阈值。
5. 可定制化进度条，终端实时显示运行结果。
6. 更漂亮的错误打印。
7. 简洁的代码可用于入门深度学习。

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
2. save ckpt时save optimizer和lr_scheduler

## License

[MIT](LICENSE) © m1dsolo
