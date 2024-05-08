import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

data_path = "deceptive-opinion.csv"
df = pd.read_csv(data_path)

class TextCleaner():
    def __init__(self):
        pass
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

cleaner = TextCleaner()
df['cleaned_text'] = df['text'].apply(cleaner.clean_text)

label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['deceptive'].tolist())

train_df, test_df = train_test_split(df, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_data(examples):
    return tokenizer(examples["cleaned_text"], truncation=True)

tokenized_train = train_dataset.map(tokenize_data, batched=True)
tokenized_test = test_dataset.map(tokenize_data, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2) 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

evaluation_results = trainer.evaluate()
print(f"Evaluation Results: {evaluation_results}")

trainer.save_model('model')

def predict_text(text):
    cleaned_text = cleaner.clean_text(text)
    tokenized_text = tokenizer(cleaned_text, truncation=True, return_tensors="pt")
    predictions = model(**tokenized_text)
    predicted_label_idx = predictions.logits.argmax(-1).item()
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
    return predicted_label

example_text = "Advertised amenities like the spa and gym were under maintenance during our entire visit, which was extremely frustrating and not communicated beforehand."
predicted_label = predict_text(example_text)
print(f"Predicted Label: {predicted_label}")



# import pandas as pd
# import re
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# data_path = "cleaned_data_with_stopwords_removed.csv"
# df = pd.read_csv(data_path)

# class TextCleaner():
#     def __init__(self):
#         pass
    
#     def clean_text(self, text):
#         text = text.lower()
#         text = re.sub(r'<.*?>', '', text)
#         text = re.sub(r'http\S+', '', text)
#         text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

# cleaner = TextCleaner()
# df['cleaned_text'] = df['text'].apply(cleaner.clean_text)

# label_encoder = preprocessing.LabelEncoder()
# df['label'] = label_encoder.fit_transform(df['deceptive'].tolist())

# train_df, test_df = train_test_split(df, test_size=0.2)

# tokenizer = AutoTokenizer.from_pretrained('model')
# model = AutoModelForSequenceClassification.from_pretrained('model', num_labels=2)

# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

# def tokenize_data(examples):
#     return tokenizer(examples["cleaned_text"], truncation=True)

# tokenized_train = train_dataset.map(tokenize_data, batched=True)
# tokenized_test = test_dataset.map(tokenize_data, batched=True)

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# def predict_text(text):
#     cleaned_text = cleaner.clean_text(text)
#     tokenized_text = tokenizer(cleaned_text, truncation=True, return_tensors="pt")
#     predictions = model(**tokenized_text)
#     predicted_label_idx = predictions.logits.argmax(-1).item()
#     predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
#     return predicted_label

# example_text = "Very disappointed with the cleanliness; the carpets were stained and the bathroom had a persistent foul odor"
# predicted_label = predict_text(example_text)
# print(f"Predicted Label: {predicted_label}")
