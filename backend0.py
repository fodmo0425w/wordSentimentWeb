import torch
import jieba
from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizerFast, BertModel, BertTokenizer
import logging
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)
import random
# 初始化Flask应用
app = Flask(__name__)
# 日志配置
log_file = "output_.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 初始化 BERT Fast Tokenizer 和模型
model_name = "bert_base_uncase"
try:
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name,ignore_mismatched_sizes=True).to(device)
    logger.info("BERT模型加载成功")
except Exception as e:
    logger.error(f"BERT模型加载失败: {e}")
    raise
def load_sentiwordnet(file_path):
    sentiwordnet = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) != 6:
                continue
            pos, id, pos_score, neg_score, synset_terms, gloss = fields

            try:
                pos_score = float(pos_score)
                neg_score = float(neg_score)
            except ValueError:
                continue

            synset_terms = synset_terms.split()
            for term in synset_terms:
                term_base = term.split("#")[0]
                if term_base not in sentiwordnet:
                    sentiwordnet[term_base] = []
                sentiwordnet[term_base].append((pos_score, neg_score))
    logger.info(f"Loaded {len(sentiwordnet)} entries from SentiWordNet.")
    return sentiwordnet
# 获取情感值函数
def get_sentiment_vector(word, sentiwordnet):
    if word not in sentiwordnet:
        return torch.zeros(3)
    pos_score, neg_score = 0.0, 0.0
    for score in sentiwordnet[word]:
        pos_score += score[0]
        neg_score += score[1]
    pos_score /= len(sentiwordnet[word])
    neg_score /= len(sentiwordnet[word])

    if pos_score > neg_score:
        return torch.tensor([1, 0, 0], dtype=torch.float32)
    elif neg_score > pos_score:
        return torch.tensor([0, 0, 1], dtype=torch.float32)
    else:
        return torch.tensor([0, 1, 0], dtype=torch.float32)
# 处理句子并生成情感向量和BERT嵌入
def process_sentence(sentence, tokenizer, bert_model, sentiwordnet):
    inputs = tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True, max_length=382
    ).to(bert_model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    offsets = inputs.pop("offset_mapping", None)

    word_map = {}
    if offsets is not None:
        original_words = sentence.split()
        word_idx = 0
        for i, (start, end) in enumerate(offsets[0].tolist()):
            if start == 0 and end == 0:
                continue
            while word_idx < len(original_words) and start >= len(
                " ".join(original_words[: word_idx + 1])
            ):
                word_idx += 1
            if word_idx < len(original_words):
                word_map[i] = original_words[word_idx].lower()

    sentiment_vectors = []
    for idx, token in enumerate(tokens):
        if idx in word_map:
            sentiment_vector = get_sentiment_vector(word_map[idx], sentiwordnet)
        else:
            sentiment_vector = torch.zeros(3)
        # 将 sentiment_vector 移动到与 BERT 嵌入相同的设备
        sentiment_vector = sentiment_vector.to(bert_model.device)
        sentiment_vectors.append(sentiment_vector)

    with torch.no_grad():
        outputs = bert_model(**inputs)
    bert_embeddings = outputs.last_hidden_state.squeeze(0)[1:-1]
    sentence_vector = []
    combined_vectors = []
    for bert_vector, sentiment_vector in zip(bert_embeddings, sentiment_vectors):
        combined_vector = torch.cat((bert_vector, sentiment_vector))
        combined_vectors.append(combined_vector)

    sentence_vector = torch.mean(torch.stack(combined_vectors), dim=0)
    return sentence_vector.cpu().numpy()
# 加载LSTM模型
def load_lstm_model(input_size, hidden_size, output_size, model_path):
    class LSTMClassifier(torch.nn.Module):
        def __init__(
            self, input_size, hidden_size, output_size, num_layers=4, dropout=0.5
        ):
            super(LSTMClassifier, self).__init__()
            self.lstm = torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.fc1 = torch.nn.Linear(hidden_size, 100)
            self.fc2 = torch.nn.Linear(100, output_size)

        def forward(self, x):
            out, _ = self.lstm(x.unsqueeze(1))
            out = out[:, -1, :]
            fc1_out = self.fc1(out)
            output = self.fc2(fc1_out)
            return output

    model = LSTMClassifier(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
# 定义句子数据作为常量
SENTENCES = [
    "I am very happy today!",
    "you are such a idiot.",
    "I feel neutral about the situation.",
]
# 主函数
def main():
    output_file = "all_vectors.npy"
    sentiwordnet_path = (
        "SentiWordNet_3.0.0.txt"
    )
    lstm_model_path = "saved_models/eng_lstm_model.pth"
    sentiwordnet = load_sentiwordnet(sentiwordnet_path)
    sentence_vectors = []
    for sentence in tqdm(SENTENCES, desc="Processing sentences"):
        combined_vectors = process_sentence(
            sentence, tokenizer, bert_model, sentiwordnet
        )
        sentence_vectors.append(combined_vectors)

    # 转换为Tensor并传递给LSTM模型
    sentence_tensors = torch.tensor(sentence_vectors, dtype=torch.float32).to(device)
    lstm_model = load_lstm_model(
        input_size=sentence_tensors.shape[1],
        hidden_size=64,
        output_size=2,
        model_path=lstm_model_path,
    )

    # 进行预测
    with torch.no_grad():
        outputs = lstm_model(sentence_tensors)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # 输出预测结果
    for i, sentence in enumerate(SENTENCES):
        logger.info(f"Sentence: {sentence}, Prediction: {predictions[i]}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'english')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        sentiwordnet= (
        "SentiWordNet_3.0.0.txt"
    )
        lstm_model_path = "saved_models/eng_lstm_model.pth"
        sentence_vector = process_sentence(text, tokenizer, bert_model, sentiwordnet)
        # 转换为Tensor并传递给LSTM模型
        sentence_tensors = torch.tensor([sentence_vector], dtype=torch.float32).to(device)
        lstm_model = load_lstm_model(
            input_size=771,
            hidden_size=64,
            output_size=2,
            model_path=lstm_model_path,
    )
        logger.info("LSTM model loaded and ready for inference.")
        # 进行预测
        with torch.no_grad():
            outputs = lstm_model(sentence_tensors)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            response = {
                'prediction': 'Aggressive' if predictions[0] == 1 else 'Not aggressive'
            }
            # 输出预测结果
            logger.info(f"Sentence: {text}, Prediction: {predictions[0]}")
            return jsonify(response), 200
        
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
