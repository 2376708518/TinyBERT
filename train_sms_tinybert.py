import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW  # 使用 PyTorch 内置 AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# ====== 配置 ======
MODEL_DIR = "./tinybert"  # 你下载的 TinyBERT
DATA_PATH = "SMSSpamCollection.txt"
OUTPUT_DIR = "./tinybert_finetuned"  # 保存微调后模型
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 1. 加载数据 ======
print("🔍 加载 SMS 数据集...")
df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
texts = df['message'].tolist()
labels = df['label'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# ====== 2. 加载 Tokenizer 和 Model ======
print("📥 加载 TinyBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
bert_model = AutoModel.from_pretrained(MODEL_DIR)


class TinyBERTForSpam(nn.Module):
    def __init__(self, bert_model, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits


# 创建模型实例
model = TinyBERTForSpam(bert_model, num_classes=2).to(DEVICE)


# ====== 3. 编码数据 ======
def encode_texts(texts, labels, max_len=MAX_LEN):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))


train_dataset = encode_texts(train_texts, train_labels)
val_dataset = encode_texts(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ====== 4. 优化器 & 调度器 ======
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ====== 5. 训练循环 ======
print(f"🚀 开始训练 (设备: {DEVICE})...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    # 验证
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())

    acc = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {acc:.4f}")

# ====== 6. 保存模型 ======
print("💾 保存微调模型...")
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
bert_model.config.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 模型已保存至: {OUTPUT_DIR}")