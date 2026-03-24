import os
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 环境与 Text-CNN 结构定义
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128


class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, kernel_sizes=[3, 4, 5], num_channels=100):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(len(kernel_sizes) * num_channels, 2)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return self.fc(self.dropout(x))


# ==========================================
# 2. 实验核心引擎
# ==========================================
def run_benchmark(model_name, model, processor, texts, labels, is_transformer=True):
    model.eval()
    preds, latencies = [], []

    # 3.4.4 预热阶段 (Warm-up)
    print(f"🚀 正在测试 {model_name}...")
    with torch.no_grad():
        for _ in range(50):
            dummy = torch.zeros((1, MAX_LEN), dtype=torch.long).to(DEVICE)
            _ = model(dummy) if not is_transformer else model(dummy)
    torch.cuda.synchronize()

    # 统计阶段 (Batch Size = 1)
    for text in texts:
        # A. 预处理分支
        if is_transformer:
            # 符合你提到的 (1)-(4) 预处理步骤
            inputs = processor(text, return_tensors="pt", padding='max_length',
                               max_length=MAX_LEN, truncation=True).to(DEVICE)
            input_ids = inputs.input_ids
            mask = inputs.attention_mask
        else:
            # Text-CNN 专用简单分词
            tokens = text.lower().split()
            ids = [processor.get(t, processor.get('<UNK>', 1)) for t in tokens]
            ids = (ids + [0] * MAX_LEN)[:MAX_LEN]
            input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
            mask = None

        # B. 推理计时
        start = time.perf_counter()
        with torch.no_grad():
            if is_transformer:
                logits = model(input_ids, attention_mask=mask).logits
            else:
                logits = model(input_ids)
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        preds.append(torch.argmax(logits, dim=1).item())

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds),
        "Recall": recall_score(labels, preds),
        "F1": f1_score(labels, preds),
        "Latency(ms)": np.mean(latencies)
    }


# ==========================================
# 3. 执行对比实验
# ==========================================
# 1. 加载测试数据
df = pd.read_csv("SMSSpamCollection.txt", sep='\t', names=['label', 'text'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
t_texts, t_labels = test_df['text'].tolist(), test_df['label'].tolist()

results = []

# 2. 加载 BERT-base
bert_tok = AutoTokenizer.from_pretrained("./bert_finetuned")
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert_finetuned").to(DEVICE)
results.append(run_benchmark("BERT-base", bert_model, bert_tok, t_texts, t_labels))

# 3. 加载 TinyBERT
tiny_tok = AutoTokenizer.from_pretrained("./tinybert_finetuned")
tiny_model = AutoModelForSequenceClassification.from_pretrained("./tinybert_finetuned").to(DEVICE)
results.append(run_benchmark("TinyBERT", tiny_model, tiny_tok, t_texts, t_labels))

# 4. 加载 Text-CNN (适配你的保存格式)
cnn_ckpt = torch.load("./cnn_finetuned/pytorch_model.bin")
cnn_model = TextCNN(vocab_size=10000).to(DEVICE)
cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
results.append(run_benchmark("Text-CNN", cnn_model, cnn_ckpt['vocab'], t_texts, t_labels, is_transformer=False))

# ==========================================
# 4. 结果可视化 (避开 Canvas 报错)
# ==========================================
res_df = pd.DataFrame(results)
print(res_df)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
# 画出 F1-Score 与 Latency 的双轴图或对比图
ax = sns.barplot(data=res_df, x="Model", y="F1", palette="Blues_d")
ax2 = ax.twinx()
sns.lineplot(data=res_df, x="Model", y="Latency(ms)", marker='o', color='red', ax=ax2)
plt.title("Performance vs Latency Comparison")
plt.savefig("final_comparison.png")
