'''
    Description: This code is named language model, which can genreate texts based on word-level
    The input of this model is the real reviews, and each output at each time is just be influenced by the previous words

    The review generation process considers the user-item interaction information and rating information
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from copy import deepcopy
from utils import config as conf
import random


class Encoder(nn.Module):
    def __init__(self, user_num, item_num):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, conf.enc_hid_dim)
        self.item_embedding = nn.Embedding(item_num, conf.enc_hid_dim)
        self.rating_embedding = nn.Embedding(conf.rating_range + 1, conf.enc_hid_dim)
        self.hidden_layer = nn.Linear(conf.enc_hid_dim * 3, conf.dec_hid_dim)

    def forward(self, user, item, rating):
        # user/item/rating shape: (batch_size)
        # embed shape: (batch_size, enc_hid_dim)
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        rating_embed = self.rating_embedding(rating)
        # concat embed shape: (batch_size, enc_hid_dim * 3)
        concat_embed = torch.cat((user_embed, item_embed, rating_embed), dim=-1)
        # hidden_state shape: (batch_size, dec_hid_dim)
        hidden_state = torch.tanh(self.hidden_layer(concat_embed)).view(-1, conf.dec_hid_dim)

        '''TODO: TEST REVIEW GENERATION'''
        # # # # START ------ ****** verify review generation with GRU ****** ####
        # Instead of generate a initial hidden state based on features, use a all-zero placeholder to test
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # hidden_state = torch.zeros(1, user.shape[0], conf.hidden_dim).to(device)
        # ****** verify review generation with GRU ****** ------ END ####

        return hidden_state, user_embed, item_embed, rating_embed


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (n_layers*n_directions, batch_size, dec_hid_dim)
        # encoder_outputs: (feature_length(=3), batch_size, enc_hid_dim)
        # batch_size = encoder_outputs.shape[1]
        feature_len = encoder_outputs.shape[0]
        # repeat decoder hidden state feature_len times
        hidden = hidden[-1].unsqueeze(1).repeat(1, feature_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, feature_len, dec hid dim]
        # encoder_outputs = [batch size, feature_len, enc hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, 1]
        attention = energy.squeeze(2)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()
        # ouptut_dim: vocabulary size of the review text
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attention = attention
        # word embedding layer
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Decoder RNN
        self.rnn = nn.GRU(emb_dim, dec_hid_dim, num_layers=self.num_layers, dropout=dropout)
        self.attention_fc = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.decodetop_fc = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        # Word output (with attention)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim, bias=False)
        # dropout for word embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input shape: (batch_size)
        # hidden shape: (n_layers*n_directions, batch_size, dec_hid_dim)
        # encoder_outputs shape: (feature_length(=3), batch_size, enc_hid_dim)
        input = input.unsqueeeze(0)                         # input shape: (1, batch_size)
        word_embed = self.dropout(self.embedding(input))    # word_embed shape: (1, batch_size, word_embed_dim)
        output, hidden = self.rnn(word_embed, hidden)       # hidden: (n_layers*n_directions, batch_size, dec_hid_dim)
        a = self.attention(hidden, encoder_outputs)         # a shape: (batch_size, feature_len(=3))
        a = a.unsqueeze(dim=1)                              # a shape: (batch_size, 1, feature_len(=3))
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # encoder_outputs: (batch_size, feature_len(=3), enc_hid_dim)
        weighted = torch.bmm(a, encoder_outputs)            # weighted: (batch_size, 1, enc_hid_dim)
        # Compute attention context vector
        attn_hidden = torch.tanh(self.attention_fc(weighted) + self.decodetop_fc(hidden[-1]))
        prediction = self.fc_out(attn_hidden)               # prediction shape: (batch_size, output_dim)

        return prediction, hidden


class Att2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, user, item, rating, text, teacher_forcing_ratio=1.0):
        # user/item/rating: (batch_size)
        # text: (text_len, batch_size)
        # teacher_forcing_ratio is probability to use teacher forcing
        batch_size = user.shape[0]
        text_length = text.shape[0]
        text_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(text_length, batch_size, text_vocab_size).to(self.device)
        hidden, user_embed, item_embed, rating_embed = self.encoder(user, item, rating)
        # compute encoder output
        encoder_outputs = torch.stack([user_embed, item_embed, rating_embed], dim=0)
        # construct initial hidden state
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        # first input to the decoder is the <sos> tokens
        input = text[0, :]

        for t in range(1, text_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = outputs
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from the prediction
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = text[t] if teacher_force else top1

        return outputs
