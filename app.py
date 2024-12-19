import torch
import jieba
from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizerFast, BertModel, BertTokenizer
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)
import random

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

########################
# 英文部分相关初始化
########################

en_model_name = "/home/eason/pdd_paper_new/troll_detection_paper/english_bert"
try:
    en_tokenizer = BertTokenizerFast.from_pretrained(en_model_name)
    en_bert_model = BertModel.from_pretrained(en_model_name).to(device)
    logger.info("英文BERT模型加载成功")
except Exception as e:
    logger.error(f"英文BERT模型加载失败: {e}")
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
            pos, sid, pos_score, neg_score, synset_terms, gloss = fields
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

en_sentiwordnet_path = "/home/eason/wordSentimentWeb/SentiWordNet_3.0.0.txt"
en_sentiwordnet = load_sentiwordnet(en_sentiwordnet_path)

def get_sentiment_vector_en(word, sentiwordnet):
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

def process_sentence_en(sentence, tokenizer, bert_model, sentiwordnet):
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
            sentiment_vector = get_sentiment_vector_en(word_map[idx], sentiwordnet)
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

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=0.5):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = out[:, -1, :]
        fc1_out = self.fc1(out)
        output = self.fc2(fc1_out)
        return output

# 预加载英文LSTM模型
en_lstm_model_path = "saved_models/eng_lstm_model.pth"
input_size_en = 1024 + 3  # BERT-large hidden_size=1024 + 3
hidden_size_en = 64
output_size_en = 2

def load_lstm_model_en(input_size, hidden_size, output_size, model_path):
    model = AdvancedLSTMModel(input_size, hidden_size, output_size, num_layers=4, dropout=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    en_lstm_model = load_lstm_model_en(
        input_size=input_size_en,
        hidden_size=hidden_size_en,
        output_size=output_size_en,
        model_path=en_lstm_model_path
    )
    logger.info("英文LSTM模型预加载完成")
except Exception as e:
    logger.error(f"英文LSTM模型加载失败: {e}")
    raise

def predict_english(text):
    sentence_vector = process_sentence_en(text, en_tokenizer, en_bert_model, en_sentiwordnet)
    logger.info(f"Sentence vector size (English): {sentence_vector.shape[0]}")
    if sentence_vector.shape[0] != 1027:
        logger.error(f"Expected sentence vector size 1027, but got {sentence_vector.shape[0]}")
        raise ValueError(f"Expected sentence vector size 1027, but got {sentence_vector.shape[0]}")
    sentence_tensors = torch.tensor([sentence_vector], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = en_lstm_model(sentence_tensors)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        prediction_label = '杠精发言' if predictions[0] == 1 else '非杠精发言'
    return prediction_label
########################
# 中文部分相关初始化
########################

zh_model_path = "/home/eason/pdd_paper_new/troll_detection_paper/chinese_bert"
try:
    zh_tokenizer = BertTokenizer.from_pretrained(zh_model_path)
    zh_bert_model = BertModel.from_pretrained(zh_model_path).to(device)
    logger.info("中文BERT模型加载成功")
except Exception as e:
    logger.error(f"中文BERT模型加载失败: {e}")
    raise

def load_sentiment_dict(file_path):
    sentiment_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            phrase = ' '.join(parts[:-1])
            score = float(parts[-1])
            sentiment_dict[phrase] = score
    logger.info(f"Loaded {len(sentiment_dict)} entries from sentiment dictionary.")
    return sentiment_dict

zh_sentiment_dict_path = "/home/eason/wordSentimentWeb/sentiment_dict.txt"
zh_sentiment_dict = load_sentiment_dict(zh_sentiment_dict_path)

def get_sentiment_vector_zh(word):
    sentiment_score = zh_sentiment_dict.get(word, None)
    if sentiment_score is None:
        return torch.zeros(3, device=device)
    if sentiment_score > 0:
        one_hot_vector = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
    elif sentiment_score < -5:
        one_hot_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    else:
        one_hot_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    return one_hot_vector

def phrase_vector_with_sentiment_zh(phrase, model, tokenizer):
    inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    bert_vector = outputs.last_hidden_state.mean(dim=1).squeeze()
    sentiment_vector = get_sentiment_vector_zh(phrase)
    combined_vector = torch.cat((bert_vector, sentiment_vector))
    return combined_vector

def sentence_to_combined_vector_zh(sentence, model, tokenizer):
    words = jieba.lcut(sentence)
    word_vectors = []
    for word in words:
        vector = phrase_vector_with_sentiment_zh(word, model, tokenizer)
        word_vectors.append(vector)

    sentence_matrix = torch.stack(word_vectors)
    sentence_vector = sentence_matrix.mean(dim=0)
    return sentence_vector.cpu().numpy()

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

zh_lstm_model_path = "/home/eason/pdd_paper_new/troll_detection_paper/saved_models/chinese_lstm_model.pth"

def load_lstm_model_zh(input_size, hidden_size, output_size, model_path):
    model = AdvancedLSTMModel(
        input_size, hidden_size, output_size, num_layers=4, dropout=0.5
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_chinese(text):
    sentence_vector = sentence_to_combined_vector_zh(text, zh_bert_model, zh_tokenizer)
    sentence_tensors = torch.tensor([sentence_vector], dtype=torch.float32).to(device)
    zh_lstm_model = load_lstm_model_zh(input_size=1027, hidden_size=64, output_size=2, model_path=zh_lstm_model_path)
    with torch.no_grad():
        outputs = zh_lstm_model(sentence_tensors)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        prediction_label = '杠精发言' if predictions[0] == 1 else '非杠精发言'
    return prediction_label

########################
# Flask 路由部分
########################

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '').strip()
        language = data.get('language', 'english')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if language == 'english':
            prediction = predict_english(text)
        elif language == 'chinese':
            prediction = predict_chinese(text)
        else:
            return jsonify({'error': 'Unsupported language'}), 400

        response = {
            'prediction': prediction
        }
        logger.info(f"Sentence: {text}, Language: {language}, Prediction: {prediction}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
