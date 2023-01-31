#!/usr/bin/env python3

import numpy as np # linear algebra
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import nltk


class LSTMAnalyzer(nn.Module):
    def __init__(self, no_layers, output_dim,
                    hidden_dim, embedding_dim, device, 
                    batch_size = 64, drop_prob=0.5 ):
        super(LSTMAnalyzer,self).__init__()

        # download stop words from nltk lib
        # used during text preprocessing
        nltk.download('stopwords')        
        
        # layer _params
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device 
        self.no_layers = no_layers      
        print(device)
        self.to(device)   
    
    def initialize_layers(self, vocab):
        print("Initializng layers")
        self.vocab_size = len(vocab) + 1

        # embedding and LSTM layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=self.no_layers, batch_first=True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()  

            
    def padder(self, sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features
    
    def preprocess_string(self, s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s

    def tokenize(self, x_train, x_val):
        word_list = []

        stop_words = set(stopwords.words('english')) 
        for sent in x_train:
            for word in sent.lower().split():
                word = self.preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)
    
        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
        
        # tockenize
        final_list_train,final_list_test = [],[]
        for sent in x_train:
                final_list_train.append([onehot_dict[self.preprocess_string(word)] for word in sent.lower().split() 
                                        if self.preprocess_string(word) in onehot_dict.keys()])
        for sent in x_val:
                final_list_test.append([onehot_dict[self.preprocess_string(word)] for word in sent.lower().split() 
                                        if self.preprocess_string(word) in onehot_dict.keys()])
                
        # encoded_train = [1 if label =='positive' else 0 for label in y_train]  
        # encoded_test = [1 if label =='positive' else 0 for label in y_val] 
        return np.array(final_list_train), np.array(final_list_test), onehot_dict

    def process_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        
        # tokenizing the smaple words dataset
        x_train_tokenized, x_test_tokenized, self.vocab = self.tokenize(self.x_train, self.x_test)
        print("Tokenized dataset")

        #we have very less number of reviews with length > 500.
        #So we will consideronly those below it.
        x_train_pad = self.padder(x_train_tokenized, 500)
        x_test_pad = self.padder(x_test_tokenized, 500)
        print("Padded dataset")

        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(self.y_train))
        valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(self.y_test))

        # make sure to SHUFFLE your data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
        print("Loaded datasets as tensors")

        # initializing nn layers
        self.initialize_layers(self.vocab)

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
        hidden = (h0,c0)
        return hidden

    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    # function to predict accuracy
    def acc(self, pred,label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item()
    
    def train_model(self):

        # loss and optimization functions
        lr=0.001

        criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        clip = 5
        epochs = 5 
        valid_loss_min = np.Inf
        # train for some number of epochs
        epoch_tr_loss,epoch_vl_loss = [],[]
        epoch_tr_acc,epoch_vl_acc = [],[]

        for epoch in range(epochs):
            train_losses = []
            train_acc = 0.0
            self.train()
            # initialize hidden state 
            h = self.init_hidden(self.batch_size)
            for inputs, labels in self.train_loader:
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)   
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])
                
                self.zero_grad()
                output, h = self.forward(inputs,h)
                
                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                train_losses.append(loss.item())
                # calculating accuracy
                accuracy = self.acc(output,labels)
                train_acc += accuracy
                #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
        
            
                
            val_h = self.init_hidden(self.batch_size)
            val_losses = []
            val_acc = 0.0
            self.eval()
            for inputs, labels in self.valid_loader:
                    val_h = tuple([each.data for each in val_h])

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    output, val_h = self.forward(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())
                    
                    accuracy = self.acc(output,labels)
                    val_acc += accuracy
                    
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_acc = train_acc/len(self.train_loader.dataset)
            epoch_val_acc = val_acc/len(self.valid_loader.dataset)
            epoch_tr_loss.append(epoch_train_loss)
            epoch_vl_loss.append(epoch_val_loss)
            epoch_tr_acc.append(epoch_train_acc)
            epoch_vl_acc.append(epoch_val_acc)
            print(f'Epoch {epoch+1}') 
            print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
            print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
            if epoch_val_loss <= valid_loss_min:
                torch.save(self.state_dict(), '../working/state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
                valid_loss_min = epoch_val_loss
            print(25*'==')
    
    def predict_text(self, text):

        word_seq = np.array([self.vocab[self.preprocess_string(word)] for word in text.split() 
                            if self.preprocess_string(word) in self.vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(self.padder(word_seq,500))
        inputs = pad.to(self.device)
        batch_size = 1
        h = self.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = self.forward(inputs, h)

        return(output.item()) 




    