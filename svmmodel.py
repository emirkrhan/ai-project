# import pandas as pd
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# data = pd.read_csv("deceptive-opinion.csv")
# data['text'] = data['text'].str.replace('\n', '')
# data['text'] = data['text'].str.strip()
# data['text'] = data['text'].str.lower()

# stop_words = set(stopwords.words("english"))

# def preprocess_text(text):
#     words = word_tokenize(text)
#     filtered_words = [word for word in words if word not in stop_words]
#     filtered_text = " ".join(filtered_words)
#     return filtered_text

# data['text'] = data['text'].apply(preprocess_text)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     return tokens

# tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=5, max_df=0.95)
# tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# data_with_tfidf = pd.concat([data, tfidf_df], axis=1)

# data_with_tfidf['word_count'] = data['text'].apply(lambda x: len(x.split()))
# data_with_tfidf['sentence_length'] = data['text'].apply(lambda x: len(x.split('.')))
# top_words = tfidf_df.sum().sort_values(ascending=False).head(10).index.tolist()
# for word in top_words:
#     data_with_tfidf['word_' + word] = data['text'].apply(lambda x: x.split().count(word))

# bottom_words = tfidf_df.sum().sort_values().head(10).index.tolist()
# for word in bottom_words:
#     data_with_tfidf['word_' + word] = data['text'].apply(lambda x: x.split().count(word))

# data_with_tfidf['longest_word_length'] = data['text'].apply(lambda x: max(len(word) for word in x.split()))

# X = data_with_tfidf.drop(['deceptive', 'hotel', 'polarity', 'source', 'text'], axis=1)
# y = data_with_tfidf['deceptive']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# class_weights = {'truthful': 1, 'deceptive': 2}  

# svm_model = SVC(kernel='linear', class_weight=class_weights, random_state=42)  
# svm_model.fit(X_train, y_train)

# y_pred = svm_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_test, y_pred)

# print("Test Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("Confusion Matrix:\n", conf_matrix)
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

data = pd.read_csv("deceptive-opinion.csv")
data['text'] = data['text'].str.replace('\n', '')
data['text'] = data['text'].str.strip()

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['text'] = data['text'].apply(preprocess_text)

def tokenize(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=5, max_df=0.95)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

data_with_tfidf = pd.concat([data, tfidf_df], axis=1)

data_with_tfidf['word_count'] = data['text'].apply(lambda x: len(x.split()))
data_with_tfidf['sentence_length'] = data['text'].apply(lambda x: len(x.split('.')))
top_words = tfidf_df.sum().sort_values(ascending=False).head(10).index.tolist()
for word in top_words:
    data_with_tfidf['word_' + word] = data['text'].apply(lambda x: x.split().count(word))

bottom_words = tfidf_df.sum().sort_values().head(10).index.tolist()
for word in bottom_words:
    data_with_tfidf['word_' + word] = data['text'].apply(lambda x: x.split().count(word))

data_with_tfidf['longest_word_length'] = data['text'].apply(lambda x: max(len(word) for word in x.split()))

X = data_with_tfidf.drop(['deceptive', 'hotel', 'polarity', 'source', 'text'], axis=1)
y = data_with_tfidf['deceptive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = {'truthful': 1, 'deceptive': 2}  

svm_model = SVC(kernel='linear', class_weight=class_weights, random_state=42)  
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

new_text = "The hotel's location was perfect, the staff extremely helpful, and the rooms spotlessly clean; I will definitely visit again!"
new_text_preprocessed = preprocess_text(new_text)
new_text_vectorized = tfidf_vectorizer.transform([new_text_preprocessed])
new_text_features = pd.DataFrame(new_text_vectorized.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
new_text_features['word_count'] = len(new_text_preprocessed.split())
new_text_features['sentence_length'] = len(new_text_preprocessed.split('.'))
new_text_features['longest_word_length'] = max(len(word) for word in new_text_preprocessed.split())

expected_features = X_train.columns
for feature in expected_features:
    if feature not in new_text_features.columns:
        new_text_features[feature] = 0
new_text_features = new_text_features.reindex(columns=expected_features)

prediction = svm_model.predict(new_text_features)
print("The text is predicted as:", "Deceptive" if prediction[0] == 'Deceptive' else "Truthful")
