from torch import optim,nn
from torchvision import models
import torch
#Sequence to be used in the sequential
def build_sequence(input_layer,output_layer,hidden_layer,dropout):
    if len(hidden_layer)>0:
        sequ=[nn.Linear(input_layer,hidden_layer[0]),nn.ReLU(),nn.Dropout(dropout[0])]
        for k in range(len(hidden_layer)-1):
            sequ=sequ
            [nn.Linear(hidden_layer[k],hidden_layer[k+1]),nn.ReLU(),nn.Dropout(dropout[k+1])]

        sequ=sequ+[nn.Linear(hidden_layer[len(hidden_layer)-1],output_layer),nn.LogSoftmax(dim=1)]

    else:
        sequ=[nn.Linear(input_layer,output_layer),nn.LogSoftmax(dim=1)]
        
    return sequ
#Model chosed by the user
def choice_model(modl):
    #Three models
    if modl=='alexnet':
        return models.alexnet(pretrained=True), 9216,'alexnet'
    elif modl=='densenet169':
        return models.densenet169(pretrained=True),1664,'densenet169'
    else:
        return models.vgg16(pretrained=True),25088,'vgg16'

#Testing the model
def test_model(Model,testloader,device,criterion):
    test_loss = 0
    test_accuracy = 0
    Model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logresult = Model.forward(inputs)
            test_loss += criterion(logresult, labels).item()
            result = torch.exp(logresult)
            testeur = torch.max(result, dim=1)[1]== labels
            test_accuracy += torch.mean(testeur.type(torch.FloatTensor)).item()
        print(f"test loss: {test_loss/len(testloader):.3f}.. "
        f"test accuracy: {test_accuracy/len(testloader):.3f}")
    return test_accuracy/len(testloader)
def train_model(epochs,dataloaders,device,criterion,optimizer,Model,current_epoch):
    for epoch in range(epochs):
        training_loss=0
        for inputs, labels in dataloaders['train']:
            optimizer.zero_grad()   
            inputs, labels = inputs.to(device), labels.to(device)        
            logresult = Model.forward(inputs)
            loss = criterion(logresult, labels)
            loss.backward()
            optimizer.step()     
            training_loss += loss.item()
        valid_loss = 0
        accuracy = 0
        Model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logresult = Model.forward(inputs)
                valid_loss += criterion(logresult, labels).item()
                result = torch.exp(logresult)
                testeur = torch.max(result, dim=1)[1]== labels
                accuracy += torch.mean(testeur.type(torch.FloatTensor)).item()

        print(f"Epoch {current_epoch+epoch+1}.. "
                f"training loss: {training_loss/len(dataloaders['train']):.3f}.. "
                f"validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                f"valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")
        
        Model.train()
    return loss,current_epoch+epoch+1,accuracy/len(dataloaders['valid'])