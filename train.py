from utility_func import *
from model_func import *
from torch import optim,nn
import argparse
parser=argparse.ArgumentParser(description='Training hyperparameters and options')
parser.add_argument('--checkpoint',type=str,default='checkpoint',type=str,help='name of checkpoint to save (default: checkpoint)')
parser.add_argument('data_directory',help='images directory')
parser.add_argument('--model',type=str,default="vgg16",help='choose model between vgg16,densenet169 or alexnet (default: vgg16)')
parser.add_argument('--output_layer',type=int,default=102,help='number of output categories')
parser.add_argument('--hidden_layer',type=int,default=[1000],nargs='*',help='output size of hidden layers (default: [1000]). Could be for instance 1000 525')
parser.add_argument('--dropout',type=float,default=[0.1],nargs='*',help='dropout (default: 0.1)')
parser.add_argument('--device',type=str,default="cuda",help='cuda or cpu (default: cuda)')
parser.add_argument('--learn_rate',type=float,default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--epochs',type=int,default=10,help='number of epochs (default: 10)')
parser.add_argument('--Pcheckpoint',type=str,default='',help='Give checkpoint name if training will resume with previous Model. e.g checkpoint_vgg169')
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#building the model
def trainning(save_path,modl,output_lay,hidden_layer,dropout,device,learn_rate,epochs,pathdir,chekpoint):
    dataloaders,class_to_idx=load_data(pathdir)
    criterion = nn.NLLLoss()  
    current_epoch=0
    name_mod=modl
    if chekpoint!='.pth':
        current_epoch,Model,optimizer,loss,criterion,input_lay,name_mod=loading_ck(chekpoint,criterion,device)
        Model.train()
    else:
        Model,input_lay,name_mod=choice_model(modl)   
        Model.classifier=nn.Sequential(*build_sequence(input_lay,output_lay,hidden_layer,dropout))   
        Model.to(device)
        optimizer = optim.Adam(Model.classifier.parameters(), lr=learn_rate)
    print("************Training session**************\n")
    loss,epoc,acc_val=train_model(epochs,dataloaders,device,criterion,optimizer,Model,current_epoch)   
    print("\n ************Testing session************** \n")
    acc_test=test_model(Model,dataloaders['test'],device,criterion)
    saving_ck(save_path,epochs+current_epoch,loss,Model.state_dict(),Model.classifier,optimizer.state_dict(),class_to_idx,criterion,input_lay,hidden_layer,output_lay,dropout,name_mod)
    print("\n ************Model Summary************ \n")
    print("Name of the Model: {}".format(name_mod))
    print("Epochs: {}".format(epoc))
    print("Accuracy on validation set: {}%".format(100*acc_val))
    print("Accuracy on test set: {}%".format(100*acc_test))
    
    
trainning(args.checkpoint + ".pth",args.model,args.output_layer,args.hidden_layer,args.dropout,device,args.learn_rate,args.epochs,args.data_directory,args.Pcheckpoint + ".pth")