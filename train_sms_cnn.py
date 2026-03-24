import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import Counter

# ====== 配置 ======
DATA_PATH = "SMSSpamCollection.txt"
OUTPUT_DIR = "./cnn_finetuned"
MAX_LEN = 128          # 最大序列长度
VOCAB_SIZE = 10000     # 词汇表大小（前 10k 高频词）
EMBED_DIM = 128        # 词向量维度
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
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

# ====== 2. 构建词汇表 & 文本编码 ======


def build_vocab(texts, vocab_size):
    word_counter = Counter()
    for text in texts:
        word_counter.update(text.lower().split())
    # 保留高频词，索引从 2 开始（0: pad, 1: unk）
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_texts(texts, vocab, max_len=MAX_LEN):
    encoded = []
    for text in texts:
        tokens = text.lower().split()
        ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
        if len(ids) < max_len:
            ids += [vocab['<PAD>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


print("📚 构建词汇表...")
vocab = build_vocab(train_texts, VOCAB_SIZE)
train_enc = encode_texts(train_texts, vocab, MAX_LEN)
val_enc = encode_texts(val_texts, vocab, MAX_LEN)

train_dataset = TensorDataset(train_enc, torch.tensor(train_labels))
val_dataset = TensorDataset(val_enc, torch.tensor(val_labels))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ====== 3. 定义 Text-CNN 模型 ======


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, kernel_sizes=[3, 4, 5], num_channels=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)

    def forward(self, input_ids):
        # input_ids: [B, L]
        x = self.embedding(input_ids).unsqueeze(1)  # [B, 1, L, D]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [B, C, L-k+1]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [B, C]
        x = torch.cat(x, dim=1)  # [B, C*len(kernels)]
        logits = self.fc(self.dropout(x))
        return logits


model = TextCNN(VOCAB_SIZE, EMBED_DIM, num_classes=2).to(DEVICE)

# ====== 4. 优化器 ======
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# ====== 5. 训练循环 ======
print(f"🚀 开始训练 Text-CNN (设备: {DEVICE})...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, labels = [b.to(DEVICE) for b in batch]

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # 验证
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = [b.to(DEVICE) for b in batch]
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())

    acc = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {acc:.4f}")

# ====== 6. 保存模型 ======
print("💾 保存 Text-CNN 模型...")
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'config': {
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'max_len': MAX_LEN
    }
}, os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
print(f"✅ 模型已保存至: {OUTPUT_DIR}")
