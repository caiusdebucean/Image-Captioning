import sys
import os
import torch
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import EncoderCNN, DecoderRNN


def clean_sentence(output):
    sentence = ''
    for idx in output:
        word = str(data_loader.dataset.vocab.idx2word[idx])
        if word != "<start>" and word != "<end>": #cleaning up
            sentence = sentence + " " + word
            sentence = sentence.strip() #get rid of the space at the beginning
    sentence = sentence[0].upper() + sentence[1:-2] + '.' #capital letter start and no space before end dot.
    return sentence


def get_prediction():
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    print(sentence)




transform_test = transforms.Compose([ 
    transforms.Resize(256),                         
    transforms.RandomCrop(224),                      
    #transforms.RandomHorizontalFlip(),               # no reason to flip
    transforms.ToTensor(),                          
    transforms.Normalize((0.485, 0.456, 0.406),     
                         (0.229, 0.224, 0.225))])


# Create the data loader.
data_loader = get_loader(transform=transform_test,    
                         mode='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

encoder_file = './weights/encoder-3.pkl' 
decoder_file = './weights/decoder-3.pkl'

embed_size = 512
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)
print(vocab_size)

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

encoder.to(device)
decoder.to(device)

image = image.to(device)
features = encoder(image).unsqueeze(1)
output = decoder.sample(features)
print('example output:', output)

#Verify everyithing is ok with the loaded model
assert (type(output)==list), "Output needs to be a Python list" 
assert all([type(x)==int for x in output]), "Output should be a list of integers." 
assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

#Test for one picture. Rerun for different results
get_prediction()