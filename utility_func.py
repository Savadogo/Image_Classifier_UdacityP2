from torchvision import datasets, transforms
from model_func import choice_model,build_sequence
import json
import numpy as np
import torch
import os
from PIL import Image
import numpy as np
def load_data(data_dir):
    #Define directories for trainning, validation and testing
    train_dir=data_dir + '/train'
    valid_dir=data_dir + '/valid'
    test_dir=data_dir + '/test'
    #Define transforms for sets
    train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    #Load the datasets with images
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    dataloader={'train':torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True),
                'test':torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True),
                'valid':torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)}
    
    return dataloader,train_data.class_to_idx

def categ_to_name(cat_path):
    return json.load(open(cat_path,'r'))
        
def saving_ck(save_path,epochs,loss,Model_dict,Model_classifier,optimizer_state,class_idx,criterion,input_layer,hidden_layer,output_layer,dropout,model_name):
    checkpoint={
        'epoch':epochs,
        'model_state_dict':Model_dict,
        'optimizer_state_dict':optimizer_state,
        'classifier':Model_classifier,
        'loss':loss,
        'class_to_idx':class_idx,
        'criterion':criterion,
        'input_layer':input_layer,
        'hidden_layer':hidden_layer,
        'output_layer':output_layer,
        'model_name':model_name,
        'dropout':dropout,
    }
    torch.save(checkpoint,save_path)
    print('Checkpoint saved')
    
def loading_ck(Fpathname,criterion,device): 
    if os.path.isfile(Fpathname):
        checkpoint=torch.load(Fpathname,map_location=lambda storage, loc: storage)
        Model,input_lay,nom_mod=choice_model(checkpoint['model_name'])
        #Model.classifier=torch.nn.Sequential(*build_sequence(checkpoint['input_layer'],checkpoint['output_layer'],checkpoint['hidden_layer'],checkpoint['dropout']))
        current_epoch=checkpoint['epoch']
        Model.class_to_idx=checkpoint['class_to_idx']
        for parametres in Model.parameters():
            parametres.requires_grad=False
        Model.classifier=checkpoint['classifier']
        Model.load_state_dict(checkpoint['model_state_dict'])
        Model.to(device)
        Optimizer = torch.optim.Adam(Model.classifier.parameters()) 
        Optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        Loss=checkpoint['loss']
        criterion=checkpoint['criterion']
        return current_epoch,Model,Optimizer,Loss,criterion,checkpoint['input_layer'],checkpoint['model_name']
        print('Checkpoint loaded')
    else:
        print('No such checkpoint')

def process_image(image):
    width,height=image.size
    image = image.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    width,height=image.size
    crop_size= ((width - 224)/2,(height - 224)/2,(width + 224)/2,(height + 224)/2)
    image = image.crop(crop_size)
    image = np.array(image)
    image=image/255
    Means_TO=np.array([0.485, 0.456, 0.406])
    Std_TO=np.array([0.229, 0.224, 0.225])
    image=(image-Means_TO)/Std_TO
    image=image.transpose(2,0,1)
    image=torch.from_numpy(image)
    image=image.float()
    return image