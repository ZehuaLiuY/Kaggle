import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from datasets import Dataset
import torch

# Load the dataset
train_path = '../../datasets/learning-agency-lab-automated-essay-scoring-2/train.csv'
test_path = '../../datasets/learning-agency-lab-automated-essay-scoring-2/test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Print unique labels
unique_labels = train_data['score'].nunique()
print(f"{unique_labels} unique labels")

# Text preprocessing
def clean_text(text):
    text = re.sub(r'\[^\w\s\]', '', text)
    text = text.lower()
    text.replace('\n', ' ')
    text.replace('&nbsp;', ' ')
    return text

train_data['full_text'] = train_data['full_text'].apply(clean_text)
test_data['full_text'] = test_data['full_text'].apply(clean_text)

# Tokenization
stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

train_data['full_text'] = train_data['full_text'].apply(tokenize)
test_data['full_text'] = test_data['full_text'].apply(tokenize)
train_data['score'] = train_data['score'] - 1

# Split the dataset
train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42)
train_df = train_df.drop(['essay_id'], axis=1)
valid_df = valid_df.drop(['essay_id'], axis=1)

# Convert to Dataset
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

for param in model.bert.parameters():
    param.requires_grad = False

for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

# Preprocess function
def preprocess_function(essay):
    encoding = tokenizer(essay['full_text'], truncation=True, padding='max_length', max_length=512)
    encoding['labels'] = essay['score']
    assert 'labels' in encoding, "Labels not found in encoding"
    assert 'input_ids' in encoding, "Input IDs not found in encoding"
    return encoding

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_function)
valid_dataset = valid_dataset.map(preprocess_function)

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(valid_dataset))

# Training arguments
training_args = TrainingArguments(
    output_dir = '../results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=None,
)

# Start training
trainer.train()

results = trainer.evaluate()
print(results)

trainer.save_model('../saved_model')
tokenizer.save_pretrained('../saved_model')

loaded_model = BertForSequenceClassification.from_pretrained('../saved_model')
loaded_tokenizer = BertTokenizer.from_pretrained('../saved_model')

test_encodings = loaded_tokenizer(test_data['full_text'].tolist(), truncation=True, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = loaded_model(**test_encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)
    predictions = predictions + 1

print(predictions)

# write predictions to file
test_data['score'] = predictions
test_data.drop('full_text', axis=1, inplace=True)
test_data.to_csv('../../datasets/learning-agency-lab-automated-essay-scoring-2/submission.csv', index=False)