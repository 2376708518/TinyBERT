import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 1. 资源准备
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class TraditionalMLPipeline:
    def __init__(self, max_features=5000):
        self.stemmer = PorterStemmer()
        base_stopwords = set(stopwords.words('english'))
        negation_words = {'not', 'no', 'never', 'nor', 'neither', 'none'}
        self.enhanced_stopwords = list(base_stopwords - negation_words)  # keep as set

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=self.enhanced_stopwords
        )

    def preprocess(self, text):
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = nltk.word_tokenize(text.lower())
        stemmed = [self.stemmer.stem(w) for w in tokens]
        return " ".join(stemmed)


# --- 实验主体流程 ---

# 1. 加载数据集
df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['label', 'message'])
y = df['label'].map({'ham': 0, 'spam': 1}).values

# 2. 预处理数据
pipeline = TraditionalMLPipeline(max_features=5000)
X_cleaned = [pipeline.preprocess(msg) for msg in df['message']]

# 3. 向量化
X_tfidf = pipeline.vectorizer.fit_transform(X_cleaned)

# 4. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 定义模型
models = {
    "Naive Bayes (NB)": MultinomialNB(),
    "SVM (Linear)": LinearSVC(class_weight='balanced', random_state=42, max_iter=1000),
    "Logistic Regression (LR)": LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000),
    "Random Forest (RF)": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 6. 训练与评估
results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

# 7. 打印结果
print(pd.DataFrame(results).T.round(4))
