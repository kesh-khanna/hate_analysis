from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(self.texts[7])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=2e-5)

BATCH_SIZE = 256
MAX_LEN = 128
EPOCHS = 1

df = preprocess()
X = list(df['text'])
y = list(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
counter = 0
'''
for epoch in range(EPOCHS):
    model.train()
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        counter += 1
        
        if (len(data_loader)//counter) > len(data_loader)/2:
            print("epoch " +str(epoch) +" halfway")
            counter = 0
        
    counter = 0
    print("epoch" +str(epoch) +"done")


model.save_pretrained('BERT-HateSpeech')
'''

dataset_test = TextDataset(X_test, y_test, tokenizer, MAX_LEN)
data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE)

model.eval()

correct_predictions = 0
total_predictions = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
   for batch in data_loader_test:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)

        predicted_labels += predicted.tolist()
        true_labels += labels.tolist()

        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

print(classification_report(true_labels,predicted_labels))


