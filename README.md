# arxiv-text-classification

## 项目简介
本项目为 Arxiv 上的文献分类任务

## 数据集介绍

数据来源：https://github.com/LiqunW/Long-document-dataset/
因为数据集太大，因此没有上传。

数据集特征如下：

| Class name | Number of documents | Average words |
|--|--|--|
| cs.AI ( Artificial Intelligence) | 2995 | 6212 |
| cs.CE (Computational Engineering) | 2505 | 5777 |
| cs.CV (Computer Vision) | 2525 | 5630 |
| cs.DS (Data Structures) | 4136 | 7439 |
| cs.IT (Information Theory ) | 3233 | 5938 |
| cs.NE (Neural and Evolutionary) | 3012 | 5856 |
| cs.PL (Programming Languages) | 2901 | 7012 |
| cs.SY (Systems and Control) | 3106 | 5948 |
| math.AC (Commutative Algebra ) | 2885 | 5984 |
| math.GR (Group Theory) | 3065 | 6642 |
| math.ST (Statistics Theory) | 6025 | 6983 |

## 项目流程

- 使用 BART 摘要模型对文本进行摘要（直接用pipeline）
- 微调 LongFormer 长文本分类预训练模型（分类结果可以用混淆矩阵展示）
- 有时间可以比较不同预训练模型的效果（macro-F1）

