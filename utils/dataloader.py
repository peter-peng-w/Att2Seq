import torch
import os
import json
import numpy as np
from collections import Counter


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, folder='data/Musical_Instruments_5', split='train', vocabulary=None):
        # Different split represent different dataset
        self.json_dir = os.path.join(folder, split, 'metadata')

        # Load JSON files.
        print('Loading %s ...' % self.json_dir, end = '')
        fdir = os.listdir(self.json_dir)
        self.metadata = [(fname[:-5], json.load(open(os.path.join(self.json_dir, fname)))) 
                     for fname in sorted(fdir) if not fname.startswith('.')]
        print(' finished')
        
        # Compute vocabulary.
        if split == 'train':
            texts = " ".join([m[1]['plot'][0] for m in self.metadata])
            word_counts = Counter(texts.lower().split(" "))
            word_counts = sorted(word_counts, key = word_counts.get, reverse = True)
            self.word2id = {w:i for (i, w) in enumerate(word_counts[:20000])}
            self.word2id['<UNK>'] = 20000
            self.word2id['<START>'] = 20000 + 1
            self.word2id['<END>'] = 20000 + 2
            self.word2id['<PAD>'] = 20000 + 3
            self.id2word = {i:w for (w, i) in self.word2id.items()}
        else:
            self.word2id = vocabulary
            self.id2word = {i:w for (w, i) in self.word2id.items()}

        # Pre-tokenizing all sentences.
        print('Tokenizing...', end = '')
        self.tokenized_plots = list()
        for i in range(0, len(self.metadata)):
            # extract the section of 'plot'
            text = self.metadata[i][1]['plot'][0]
            encoded_text = self.tokenize(text)
            self.tokenized_plots.append(encoded_text)
        print(' finished')
            
    def __getitem__(self, index: int):
        _, movie_data = self.metadata[index]
        text = self.tokenized_plots[index]
        return text

    def get_metadata(self, index: int):
        _, movie_data = self.metadata[index]
        return movie_data

    def tokenize(self, text):
        text = text.lower().split(" ")
        encoded_text = [self.word2id.get(w, self.word2id['<UNK>']) for w in text]
        encoded_text = encoded_text[:254]                                   # totally 254 tokens
        encoded_text_array = np.ones(256) * self.word2id['<PAD>']
        encoded_text_array[0] = self.word2id['<START>']
        encoded_text_array[1:len(encoded_text) + 1] = encoded_text
        encoded_text_array[len(encoded_text) + 1] = self.word2id['<END>']
        return torch.tensor(encoded_text_array, dtype = torch.long)

    def untokenize(self, token_ids):
        return [self.id2word[id] for id in token_ids.detach().numpy()]

    def __len__(self):
        return len(self.metadata)
