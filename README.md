# NQG-LSTM for 天池中医文献问题生成任务

源码结构来自于论文： [Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks](https://www.aclweb.org/anthology/D18-1424)

源代码来自于github: https://github.com/seanie12/neural-question-generation

数据来源：https://tianchi.aliyun.com/competition/entrance/531826/information

预训练模型MTBERT：https://code.ihub.org.cn/projects/1775



## Dependencies

This code is written in Python. Dependencies include

- python >= 3.6

- pytorch >= 1.4

- nltk

- tqdm

- [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)

  

## 改进

使用MTBERT的embedding层作为LSTM之前的embedding层，使用专业医药领域预训练数据替代了原来的glove

由于答案在原文里，所以增加了对答案的识别，延续了answer_tag的结构

由于数据集质量不好，手动做了一些数据处理



## Configuration

python -W ignore main.py [--train] [--model_path] 



## Results

初赛结果：0.5415