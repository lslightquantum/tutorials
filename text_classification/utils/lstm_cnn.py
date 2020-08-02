import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


def get_vocabulary(data, alphabet_only=True):
    vocab = set()
    for i in range(len(data)):
        tokens = []
        if alphabet_only:
            tokens = re.sub('[^a-zA-Z]', ' ', data[i]).split()
        else:
            tokens = data[i][1].split()
        vocab.update(tokens)
    return list(vocab)


def get_reduced_embedding_matrix(pretrained_words, pretrained_vectors, dataset_vocab, 
                                 add_unknown=True, add_padding=True):
    word2idx = {word: index for index, word in enumerate(pretrained_words)}
    dataset_vocab = list(dataset_vocab)
    word_idx = []
    i = 0
    while i < len(dataset_vocab):
        if dataset_vocab[i] in word2idx:
            word_idx.append(word2idx[dataset_vocab[i]])
            i += 1
        else:
            dataset_vocab.remove(dataset_vocab[i])
    word_vectors = torch.tensor([pretrained_vectors[idx] for idx in word_idx])

    if add_unknown:
        unk_word_vector = word_vectors.mean(0).unsqueeze(0)
        dataset_vocab.append('<UNK_WORD>')
        word_vectors = torch.cat((word_vectors, unk_word_vector))

    if add_padding:
        pad_word_vector = torch.rand(1, 300)
        dataset_vocab.append('<PAD_WORD>')
        word_vectors = torch.cat((word_vectors, pad_word_vector))
    
    word2idx = {word: index for index, word in enumerate(dataset_vocab)}
    return word_vectors, word2idx


class Tokenizer():
    def __init__(self, token2idx, alphabet_only=True):
        self.token2idx = token2idx
        self.alphabet_only = alphabet_only
        
    def encode(self, string, pad_to_max=True, max_len=0, return_mask=True, truncation=True):
        if self.alphabet_only:
            tokens = re.sub('[^a-zA-Z]', ' ', string).split()
        else:
            tokens = string.split()
        encode = [self.token2idx[word] if word in self.token2idx else self.token2idx['<UNK_WORD>'] for word in tokens]
        mask = []
        
        if truncation and len(encode) > max_len:
            encode = encode[:max_len]
        
        if pad_to_max:
            if return_mask:
                mask = [1]*len(encode) + [0]*(max_len - len(encode))
            if len(encode) < max_len:
                encode = encode + [self.token2idx['<PAD_WORD>']] * (max_len - len(encode))
                
        retval = {'encode': encode}
        if mask:
            retval['mask'] = mask
            
        return retval
                
    def batch_encode(self, strings, pad_to_max=False, mask=True):
        encodes = []
        masks = []
        for string in strings:
            if self.alphabet_only:
                tokens = re.sub('[^a-zA-Z]', ' ', string).split()
            else:
                tokens = string.split()
            indices = [self.token2idx[word] if word in self.token2idx else self.token2idx['<UNK_WORD>'] for word in tokens]
            encodes.append(indices)
        
        max_len = max([len(encode) for encode in encodes])

        if pad_to_max:
            for i in range(len(encodes)):
                if mask:
                    masks.append([1]*len(encodes[i]) + [0]*(max_len - len(encodes[i])))
                if len(encodes[i]) < max_len:
                    encodes[i] = encodes[i] + [self.token2idx['<PAD_WORD>']] * (max_len - len(encodes[i]))
                             
        retval = {'encodes': encodes}
        if mask:
            retval['masks'] = masks
            
        return retval


def remove_null_data(sentences, labels, tokenizer):
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        encodes = tokenizer.encode(sentence, truncation=False, pad_to_max=False, return_mask=False)
        if len(encodes['encode']) == 0:
            del sentences[i]
            del labels[i]
        else:
            i += 1


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
        
        encoding = self.tokenizer.encode(sentence, max_len=self.max_len, 
                                         truncation=True, pad_to_max=True, return_mask=True)
        
        return {'inputs': torch.tensor(encoding['encode']), 
                'masks': torch.tensor(encoding['mask']),
                'labels': torch.tensor(label, dtype=torch.long)}
    
def get_data_loader(X, y, tokenizer, max_len, batch_size):
    dataset = MyDataset(X, y, tokenizer, max_len)
    data_loader = DataLoader(dataset, num_workers=0, batch_size=batch_size)
    
    return data_loader


def accuracy(outputs, labels):
    return np.mean(labels == np.argmax(outputs, axis=1))


def train(model, criterion, accuracy, optimizer, data_loader):
    model.train()
    train_loss, train_acc = float('inf'), 0.
    losses, accuracies = [], []
    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, disable=True)
    summary_every = len(data_loader)//10
    for i, data in tqdm_iterator:
        inputs = data['inputs'].cuda()
        masks = data['masks'].cuda()
        labels = data['labels'].cuda()
        outputs = model(inputs, masks)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        outputs, labels = outputs.data.cpu().numpy(), labels.data.cpu().numpy()
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
    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, disable=True)
    with torch.no_grad():
        for i, data in tqdm_iterator:
            inputs = data['inputs'].cuda()
            masks = data['masks'].cuda()
            labels = data['labels'].cuda()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            outputs, labels = outputs.data.cpu().numpy(), labels.data.cpu().numpy()
            acc = accuracy(outputs, labels)
            losses.append(loss.item())
            accuracies.append(acc)
    eval_loss = np.mean(losses)
    eval_acc = np.mean(accuracies)
    
    return float(eval_loss), float(eval_acc)

def train_and_evaluate(model, criterion, accuracy, optimizer,
                       X_train, y_train, X_val, y_val, tokenizer, max_len, batch_size, n_epochs):
    best_eval_loss = float('inf')
    best_epochs = 0
    summary = {'train': {'loss': [], 'acc': []}, 
               'val': {'loss': [], 'acc': []}}
    print('start training: ')
    train_data_loader = get_data_loader(X_train, y_train, tokenizer, max_len, batch_size)
    eval_data_loader = get_data_loader(X_val, y_val, tokenizer, max_len, batch_size)
    for i in range(1, n_epochs + 1):
        print('-'*50)
        print('epoch ', i, ':')
        train_loss, train_acc = train(model, criterion, accuracy, optimizer, train_data_loader)
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

def test(model, criterion, accuracy, X_test, y_test, tokenizer, max_len, batch_size):
    test_data_loader = get_data_loader(X_test, y_test, tokenizer, max_len, batch_size)
    test_loss, test_acc = evaluate(model, criterion, accuracy, test_data_loader)

    print('test loss: {:.4f}; test acc: {:.4f}'.format(test_loss, test_acc))

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