# Arxiv 文献分类任务

## 项目简介
本项目为 Arxiv 上的文献分类任务

## 数据集介绍

数据来源：https://github.com/LiqunW/Long-document-dataset/
因为数据集太大，因此没有上传。

数据集特征如下：

| 类别名称 | 文档数量 | 平均字数 |
|--|--|--|
| cs.AI（人工智能） | 2995 | 6212 |
| cs.CE（计算工程） | 2505 | 5777 |
| cs.CV（计算机视觉） | 2525 | 5630 |
| cs.DS（数据结构） | 4136 | 7439 |
| cs.IT（信息理论） | 3233 | 5938 |
| cs.NE（神经与进化） | 3012 | 5856 |
| cs.PL（编程语言） | 2901 | 7012 |
| cs.SY（系统与控制） | 3106 | 5948 |
| math.AC（交换代数） | 2885 | 5984 |
| math.GR（群论） | 3065 | 6642 |
| math.ST（统计理论） | 6025 | 6983 |

## 项目流程

- 使用 BART 摘要模型对文本进行摘要（直接用pipeline）
- 微调 LongFormer 长文本分类预训练模型（分类结果可以用混淆矩阵展示）
- 有时间可以比较不同预训练模型的效果（macro-F1）

## 项目结构

- abstract: 使用 TF-IDF 和 TextRank 算法生成最后的模型输入
- classification: 使用 LongFormer，BERT，RoBERTa 进行文本分类
