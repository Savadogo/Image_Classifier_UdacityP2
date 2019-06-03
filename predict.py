from utility_func import *
from PIL import Image
import numpy as np
from torch import optim,nn

import argparse
parser=argparse.ArgumentParser(description='Prediction options')
parser.add_argument('image_path',type=str,help='images path')
parser.add_argument('--device',type=str,default="cuda",help='cuda or cpu (default: cuda)')
parser.add_argument('path_chk',type=str,help='checkpoint path')
parser.add_argument('--top_k',type=int,default=5,help='number of most likely classes(default: 5)')
parser.add_argument('--cat_to_name',type=str,default='',help='json file giving corresponding name for classes')
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
def predict(image_path, model, topk,device,cat_to_name_path):
    model.eval()
    image_prepo=process_image(Image.open(image_path)).to(device)
    dictio={v: k for k, v in model.class_to_idx.items()}
    with torch.no_grad():
        logresult = model.forward(image_prepo.unsqueeze(0))
        result=torch.exp(logresult)
        legends=[dictio[x] for x in np.array(torch.topk(result,topk,dim=1)[1][0])]
        if cat_to_name_path!=".json":
            cat_to_name=categ_to_name(cat_to_name_path)
            legends=[cat_to_name[x] for x in legends]
 
        #legends=legends[::-1]
        proba=[x for x in np.array(torch.topk(result,topk,dim=1)[0][0])]
        #proba=proba[::-1]
        longuer=max(len(l) for l in legends)+5
        print("\nThe top {} classes and their probability :\n".format(topk))
        for x in range(topk):
            print("{} : {:.4%}".format(legends[x].ljust(longuer),proba[x]))
       
def Make_a_pred(image_path,ch_path,topk,device,cat_to_name):
    criterion = nn.NLLLoss()                                       
    current_epoch,Model,optimizer,loss,criterion,input_lay,name_mod=loading_ck(ch_path,criterion,device)
    Model.to(device)
    predict(image_path,Model,topk,device,cat_to_name)
    
Make_a_pred(args.image_path,args.path_chk + ".pth",args.top_k,args.device,args.cat_to_name + ".json")
    
    