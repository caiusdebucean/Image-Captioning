import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size, momentum=0.01) #1d since its on flattened vector. default momentum is 0.1, used for running_mean and running_var computation

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1) #
        features = self.embed(features)
        features = self.batch_norm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.num_layers  = num_layers

        #LSTM for Speech Tagging is a good "landmark" for writing this
        # embedding layer - words to specified size vector
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True) #no Dropout for now

        self.hidden2tag = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        
        
        words = self.word_embeddings(captions[:,:-1]) #Since I am concatenating on that dimention, it is size mismatch unless I cut something + I don't know if I need to specify the <end> symbol, so we're cutting it, as shown in the image
        
        #print (words.shape) 
        #print(features.shape) #features have too little dimentions
        
        features = features.unsqueeze(1) #guess this is the equivalent to np.expand_dim
        #the features are already flattened from Encoder
        input_concat = torch.cat((features, words), 1) #tuple of them to the cat function
        #print(input_concat.shape)
        
        lstm_out, (hidden, cell) = self.lstm(input_concat)
        out = self.hidden2tag(lstm_out)
        
        return out
        
    def sample(self, inputs, states=None, max_len=20):
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        self.h = torch.zeros((self.num_layers, inputs.shape[0], self.hidden_size)).to(device) #from speech notebook - (self.n_layers, n_seqs, self.n_hidden) = (1, n_batch, hidden size) - also, because of CUDA errors moving these to device seems to have corrected them
        self.c = torch.zeros((self.num_layers, inputs.shape[0], self.hidden_size)).to(device) #same for cell state
        
        
        output = []
        

        while 1>0:
            lstm_out, (self.h, self.c) = self.lstm(inputs, (self.h, self.c)) #auto update the hidden and cell state

            words = self.hidden2tag(lstm_out.squeeze(1))
            
            values, indices = torch.max(words, dim=1) #we want to get the indexes where there are those values
            #print(type(indices.cpu().numpy().shape)) 
            output.append(indices.cpu().numpy().item()) #can't convert CUDA tensor to numpy. Use Tensor.cpu() - also, item() returns python scalars

            if(indices == 1 or len(output) >= max_len): #if its <end> or max len, break out of nearest loop
                break

            inputs = self.word_embeddings(indices) 
            inputs = inputs.unsqueeze(1) 
        return output