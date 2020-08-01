import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
#from torch.cuda.amp import autocast
#from torch.cuda.amp import GradScaler


class DataManager():
    def __init__(self, sentences, labels):
        category_counter = Counter()
        for label in labels:
            category_counter[label] += 1
        self.category_counts = category_counter.most_common()
        self.sentence_lens = []
        for sentence in sentences:
            self.sentence_lens.append(len(sentence))
    
    def plot_category_frequency(self):
        plt.figure(figsize=(15,5))
        x, y = zip(*self.category_counts)
        plt.bar(x, y)
        plt.xticks(rotation='vertical')
        plt.show(self)
        
    def plot_sentence_lens_dist():
        plt.figure(figsize=(15,5))
        plt.hist(sentence_lens, bins=50)
        plt.show()
    
    def get_data_by_frequency(self, sentences, labels, n=3):
        category_set = set()
        for i in range(n):
            category_set.add(self.category_counts[i][0])
        i = 0
        new_sentences = []
        new_labels = []
        for i in range(len(labels)):
            if labels[i] in category_set:
                new_sentences.append(sentences[i])
                new_labels.append(labels[i])
        
        cat2idx = {k: v for v, k in enumerate(category_set)}
        
        return new_sentences, new_labels, cat2idx


class MyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self,):
        return len(self.sentences)
    
    def __getitem__(self, item):
        sentence = self.sentences[item]
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(sentence, max_length=self.max_len, 
                                              truncation=True, return_token_type_ids=False, 
                                              padding='max_length', return_tensors='pt')
        
        return {'input_ids': encoding['input_ids'].flatten(), 
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)}
    

def get_data_loader(X, y, tokenizer, max_len, batch_size):
    dataset = MyDataset(X, y, tokenizer, max_len)
    data_loader = DataLoader(dataset, num_workers=0, batch_size=batch_size)
    
    return data_loader


def accuracy(outputs, labels):
    return np.mean(labels == np.argmax(outputs, axis=1))


def train(model, criterion, accuracy, optimizer, scheduler, data_loader):
    model.train()
    train_loss, train_acc = float('inf'), 0.
    losses, accuracies = [], []
    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    summary_every = len(data_loader)//10
    for i, data in tqdm_iterator:
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()

        ###
        #with autocast():
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs[0], labels)

        optimizer.zero_grad()

        loss.backward()
        ###
        #scaler.scale(loss).backward()
        
        optimizer.step()
        ###
        #scaler.step(optimizer)
        
        scheduler.step()
        ###
        #scaler.update()
        losses.append(loss.item())
        outputs, labels = outputs[0].data.cpu().numpy(), labels.data.cpu().numpy()
        acc = accuracy(outputs, labels)
        accuracies.append(acc)
        
        if i % summary_every == 0:
            train_loss = float(np.mean(losses))
            train_acc = float(np.mean(accuracies))
            tqdm_iterator.set_postfix(loss='{:.4f}'.format(train_loss), acc='{:.4f}'.format(train_acc))
            losses = []
            accuraies = []
    return float(train_loss), float(train_acc)
        
def evaluate(model, criterion, accuracy, data_loader):
    model.eval()
    losses, accuracies = [], []
    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    with torch.no_grad():
        for i, data in tqdm_iterator:
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            labels = data['labels'].cuda()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs[0], labels)
            outputs, labels = outputs[0].data.cpu().numpy(), labels.data.cpu().numpy()
            acc = accuracy(outputs, labels)
            losses.append(loss.item())
            accuracies.append(acc)
    eval_loss = np.mean(losses)
    eval_acc = np.mean(accuracies)
    
    return float(eval_loss), float(eval_acc)


def test(model, criterion, accuracy, X_test, y_test, tokenizer, max_len, batch_size):
    test_data_loader = get_data_loader(X_test, y_test, tokenizer, max_len, batch_size)
    test_loss, test_acc = evaluate(model, criterion, accuracy, test_data_loader)

    print('test loss: {:.4f}; test acc: {:.4f}'.format(test_loss, test_acc))


def train_and_evaluate(model, criterion, accuracy, optimizer,
                       X_train, y_train, X_val, y_val, tokenizer, max_len, batch_size, n_epochs):
    best_eval_loss = float('inf')
    best_epochs = 0
    summary = {'train': {'loss': [], 'acc': []}, 
               'val': {'loss': [], 'acc': []}}
    print('start training: ')
    train_data_loader = get_data_loader(X_train, y_train, tokenizer, max_len, batch_size)
    eval_data_loader = get_data_loader(X_val, y_val, tokenizer, max_len, batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                num_training_steps=len(train_data_loader)*n_epochs)
    #scaler = GradScaler()
    for i in range(1, n_epochs + 1):
        print('-'*50)
        print('epoch ', i, ':')
        train_loss, train_acc = train(model, criterion, accuracy, optimizer, scheduler, train_data_loader)
        summary['train']['loss'].append(train_loss)
        summary['train']['acc'].append(train_acc)
        print('train loss: {:.4f}; train acc: {:.4f}'.format(train_loss, train_acc))
        
        eval_loss, eval_acc = evaluate(model, criterion, accuracy, eval_data_loader)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epochs = i
        summary['val']['loss'].append(eval_loss)
        summary['val']['acc'].append(eval_acc)
        print('eval loss: {:.4f}; eval acc: {:.4f}'.format(eval_loss, eval_acc))
    print('-'*50)
    print('finish training.')
    
    summary['best_eval_loss'] = best_eval_loss
    summary['best_epochs'] = best_epochs
        
    return summary


def plot_summary(summary):
    data_len = len(summary['train']['loss'])

    plt.figure(figsize=(15,5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(121)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.plot(np.arange(1, data_len+1), summary['train']['loss'],  label='train')
    plt.plot(np.arange(1, data_len+1), summary['val']['loss'], label='val')
    plt.legend()
    plt.xlabel('number of epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)

    plt.subplot(122)
    plt.plot(np.arange(1, data_len+1), summary['train']['acc'],  label='train')
    plt.plot(np.arange(1, data_len+1), summary['val']['acc'], label='val')
    plt.legend()
    plt.xlabel('number of epochs', fontsize=12)
    plt.ylabel('acc', fontsize=12)

    plt.show()