import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from loader import load, preprocess
import torch.nn.functional as F
import random


class WSDDataset(Dataset) :
    def __init__(self, instances_dict, word_to_idx, keys_to_class):
        self.keys = list(instances_dict.keys())
        self.instances = [instances_dict[key] for key in self.keys]
        self.classes = [keys_to_class[key] for key in self.keys]
        self.word_to_idx = word_to_idx
        self.max_len = max(len(instance.context) for instance in self.instances)

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        instance = self.instances[index]
        #basic word embedding context
        context_index = [self.word_to_idx[word] 
             if word in self.word_to_idx 
             else self.word_to_idx["<UNK>"] for word in instance.context] 
        
        context_index =  self.pad_context(context_index) + [word_to_idx[instance.lemma]]

        return torch.tensor(context_index,dtype = torch.long), self.classes[index]
    
    def pad_context(self, context):
        if len(context) == self.max_len:
            return context
        
        pad_idx = self.word_to_idx["<PAD>"]
        context = context + [pad_idx] * (self.max_len - len(context))
        return context
    
    def create_synset_to_class_map(keys):
        key_to_class = {}
        synset_to_class = {}
        for key,senses in keys.items():
                if senses[0] not in synset_to_class:
                    synset_to_class[senses[0]] = len(synset_to_class)
                key_to_class[key] = synset_to_class[senses[0]]

        return key_to_class, synset_to_class


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim,hidden_dim, synset_size, lstm_layers = 2):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, synset_size)

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)

        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores
    
class LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim,hidden_dim, synset_size, lstm_layers = 2, num_heads = 2):
        super(LSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=False)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.hidden2tag = nn.Linear(hidden_dim, synset_size)

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        attn_out, _ = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        #attn_out = attn_out.mean(dim=1)
        tag_space = self.hidden2tag(attn_out[:,-1,:])
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores
    
def split_test(test_instances, test_keys):
    keys = list(test_instances.keys())
    random.shuffle(keys)
    train_size = int(len(keys) * 0.70)
    train_keys_list, test_keys_list = keys[:train_size], keys[train_size:]

    # Quick creation of the split dictionaries
    train_instances = {k: test_instances[k] for k in train_keys_list}
    train_keys = {k: test_keys[k] for k in train_keys_list}
    test_instances = {k: test_instances[k] for k in test_keys_list}
    test_keys_split = {k: test_keys[k] for k in test_keys_list}

    return train_instances, train_keys, test_instances, test_keys_split

def build_vocab(wsd_instances):
    word_to_idx = {}
    for instance in wsd_instances.values():
        for word in instance.context:
            if word not in word_to_idx:
                #Unique number for every word
                word_to_idx[word] = len(word_to_idx)
        if instance.lemma not in word_to_idx:
            word_to_idx[instance.lemma] = len(word_to_idx) - 1
    
    word_to_idx["<UNK>"] = len(word_to_idx) - 1
    word_to_idx["<PAD>"] = len(word_to_idx) - 1
    
    return word_to_idx



if __name__ == "__main__" : 
    dev_instances, test_instances, dev_key, test_key = load()
    train_instances, train_keys, test_instances, test_key = split_test(test_instances, test_key)
    #Dev set
    key_to_class, synset_to_class = WSDDataset.create_synset_to_class_map(dev_key)

    vocab = set()
    lemmas = set()
    for key, value in train_instances.items():
        lemmas.add(value.lemma)

    test_instances_2 = {}
    test_key_to_class = {}

    #save the instances that appear in training
    #Test set
    #SAves lemmas that appear in dev and their synset
    for key, value in test_instances.items():
        #preprocess(value)
        if value.lemma not in lemmas or test_key[key][0] not in synset_to_class:
            continue
        test_instances_2[key] = value
        test_key_to_class[key] = synset_to_class[test_key[key][0]]

    #Training
    vocab_size = len(vocab) + 2
    embedding_dim = 128 

    hidden_dim_1 = 256 
    synset_size = len(synset_to_class)  

    word_to_idx = build_vocab(dev_instances)
    train_dataset = WSDDataset(dev_instances, word_to_idx, key_to_class)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = WSDDataset(test_instances_2, word_to_idx, test_key_to_class)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    model_lstm = LSTM_Model(len(word_to_idx), embedding_dim, hidden_dim_1, synset_size, 8)
    model_lstm_attention = LSTM_Attention(len(word_to_idx), embedding_dim, hidden_dim_1, synset_size, 2, 2)
    loss_function = nn.CrossEntropyLoss()

    model = model_lstm_attention
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)    
    
    #epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0

        for context, target in train_loader:
            print(target.size())
            optimizer.zero_grad()
            tag_scores = model(context)
            loss = loss_function(tag_scores, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        #Average loss
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")


    #testing
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for context, target in test_loader:
            tag_scores = model(context)
            _, predicted = torch.max(tag_scores, 1)
            
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy}")


    
    




