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
    
    
    
    
    
class EmbeddingNetImage_V2(nn.Module):
    def __init__(self, weights, resnet_type='V1', dim_out_fc = 'as_image'):   # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage_V2, self).__init__()
        
        if resnet_type == 'V1':
            # Load the Faster R-CNN model with ResNet-50 backbone
            self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        elif resnet_type == 'V2':
            self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

        self.backbone = nn.Sequential(*list(self.faster_rcnn.backbone.children())[:-1])

        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            print(name, param.requires_grad)

        # Replace the box predictor with a custom Fast R-CNN predictor
        in_features = 3840 #self.faster_rcnn.roi_heads.box_head.fc7.in_features  # 2048

        if dim_out_fc == 'as_image':
            # Define the fully connected layers for embedding
            self.fc = nn.Linear(in_features, in_features)
        elif dim_out_fc == 'as_text':
            self.fc = nn.Linear(in_features, 1000)
            

    def forward(self, x):
        output = self.backbone(x)
        
        tensor_list = []
        for key, value in output.items():
            tensor_list.append(value.reshape(value.shape[0], value.shape[1], -1).max(dim=-1)[0])

        output = torch.cat(tensor_list, dim=1)

        output = self.fc(output) 
        
        return output



class EmbeddingNetText_V2(nn.Module):
    def __init__(self, weights, device, type_textnet='FastText', dim_out_fc = 'as_image'):  # type = 'FastText' or 'BERT'
        super(EmbeddingNetText_V2, self).__init__()
        self.device = device
        self.type_textnet = type_textnet
        
        self.model = fasttext.load_model(weights)
        
        if dim_out_fc == 'as_image':
            self.fc = nn.Linear(300, 3840)
        elif dim_out_fc == 'as_text':
            self.fc = nn.Linear(300, 1000)

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