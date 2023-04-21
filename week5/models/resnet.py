import math

import torch
import torch.nn as nn
import torchvision
import fasttext

class EmbeddingNetImage(nn.Module):
    def __init__(self, weights, resnet_type='101', dim_out_fc = 'as_image'):   # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage, self).__init__()
        
        if resnet_type == '101':
            # Load the Faster R-CNN model with ResNet-50 backbone
            self.resnet = torchvision.models.resnet101(pretrained=True)
        elif resnet_type == '50':
            self.resnet = torchvision.models.resnet50(pretrained=True)

        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity()

        if dim_out_fc == 'as_image':
            # Define the fully connected layers for embedding
            self.fc = nn.Linear(in_features, in_features)
        elif dim_out_fc == 'as_text':
            self.fc = nn.Linear(in_features, 1000)
            

    def forward(self, x):
        output = self.resnet(x)
        output = self.fc(output) 
        
        return output   
    

class EmbeddingNetText(nn.Module):
    def __init__(self, weights, device, type_textnet='FastText', dim_out_fc = 'as_image'):  # type = 'FastText' or 'BERT'
        super(EmbeddingNetText, self).__init__()
        self.device = device
        self.type_textnet = type_textnet
        
        self.model = fasttext.load_model(weights)

        self.lstm = nn.LSTM(input_size=300, hidden_size=300, num_layers=2, batch_first=True)
        
        if dim_out_fc == 'as_image':
            self.fc = nn.Sequential(nn.Linear(300, 1024), 
                                nn.PReLU(), 
                                nn.Linear(1024, 2048))
        elif dim_out_fc == 'as_text':
            self.fc = nn.Sequential(nn.Linear(300, 512), 
                                nn.PReLU(), 
                                nn.Linear(512, 1000))
            

    def forward(self, x):
        if self.type_textnet == 'FastText':
            x = [caption.replace('.', '').replace(',','').lower().split() for caption in x]
        
        output = []
        for caption in x:
            capt_output = [torch.tensor(self.model[word]).to(self.device) for word in caption]
            output.append(torch.stack(capt_output).mean(dim=0))
        output = torch.stack(output)
        
        output = self.fc(output) 
        
        return output