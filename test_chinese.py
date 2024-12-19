import torch
import jieba
from transformers import BertTokenizer, BertModel
import numpy as np
import os
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
import logging
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载情感词典
def load_sentiment_dict(file_path):
    sentiment_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            phrase, score = line.strip().split()
            sentiment_dict[phrase] = float(score)
    return sentiment_dict


sentiment_dict_path = (
    "sentiment_dict.txt"
)
sentiment_dict = load_sentiment_dict(sentiment_dict_path)


# 获取情感向量
def get_sentiment_vector(phrase):
    sentiment_score = sentiment_dict.get(phrase, None)
    if sentiment_score is None:
        return torch.zeros(3, device=device)  # 在 GPU 上创建全零向量

    if sentiment_score > 0:
        one_hot_vector = torch.tensor(
            [1, 0, 0], dtype=torch.float32, device=device
        )  # 正面情感
    elif sentiment_score < -5:
        one_hot_vector = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=device
        )  # 负面情感
    else:
        one_hot_vector = torch.tensor(
            [0, 1, 0], dtype=torch.float32, device=device
        )  # 中性情感

    return one_hot_vector

# 加载 BERT 模型
model_path = "chinese_bert"
tokenizer = BertTokenizer.from_pretrained(model_path,ignore_mismatched_sizes=True)
bert_model = BertModel.from_pretrained(model_path).to(device)  # 将模型迁移到 GPU
print("BERT 模型加载成功")


# 文本转为 BERT 输入格式
def text_to_bert_input(text, tokenizer, max_length=128):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return inputs.to(device)


# 获取短语的 BERT 向量
def phrase_vector_with_sentiment(phrase, model, tokenizer):
    inputs = text_to_bert_input(phrase, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    bert_vector = outputs.last_hidden_state.mean(
        dim=1
    ).squeeze()  # 获取短语的 BERT 表示
    sentiment_vector = get_sentiment_vector(phrase)  # 获取情感向量
    combined_vector = torch.cat((bert_vector, sentiment_vector))  # 拼接 BERT 和情感向量
    return combined_vector


# 处理句子，将所有短语的组合向量拼接成句向量
def sentence_to_combined_vector(sentence, model, tokenizer):
    words = jieba.lcut(sentence)  # 使用 jieba 分词
    word_vectors = []

    for word in words:
        vector = phrase_vector_with_sentiment(word, model, tokenizer)
        word_vectors.append(vector)

    sentence_matrix = torch.stack(word_vectors)  # 将每个词的组合向量拼接成矩阵
    sentence_vector = sentence_matrix.mean(dim=0)  # 使用均值池化得到句子向量
    return sentence_vector


# 示例句子
sentence = "那你无敌了，东西写成这样也在这里叫唤。"

# 获取句子向量
sentence_vector = sentence_to_combined_vector(sentence, bert_model, tokenizer)
# 打印结果
print("句子向量:")
print(sentence_vector.cpu().numpy())

# 加载已有模型权重
model_path = "saved_models/chinese_lstm_model.pth"


class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = out[:, -1, :]
        out = self.batch_norm(out)
        fc1_out = self.fc1(out)
        output = self.fc2(fc1_out)
        return output


# 模型参数
input_size = 771
#sentence_vector.shape[0]
hidden_size = 64
output_size = 2
model = AdvancedLSTMModel(
    input_size, hidden_size, output_size, num_layers=4, dropout=0.5
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("LSTM 模型加载成功")

# 对句子向量进行预测
with torch.no_grad():
    sentence_vector = sentence_vector.unsqueeze(0).to(device)  # 添加批次维度
    output = model(sentence_vector)
    _, prediction = torch.max(output, 1)
    print("预测结果:", prediction.item())
