# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from albumentations import *
from albumentations.pytorch import ToTensor
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian

import random
import PIL
import matplotlib.pyplot as plt
import cv2

import numpy as np
import sys, glob

from tensorboardX import SummaryWriter
from torchsummary import summary

import scipy

import time
import math
import tables

import random
import argparse

from sklearn.metrics import confusion_matrix

%load_ext tensorboard


# %%
dataname="blurry_classification"
gpuid=0

# --- densenet params
#these parameters get fed directly into the densenet class, and more description of them can be discovered there
in_channels= 3  #input channel of the data, RGB = 3


growth_rate=16 #change from 32 
block_config=(2, 2, 2, 2)
num_init_features=64
bn_size=4
drop_rate=0

parser = argparse.ArgumentParser(description='train the model')


parser.add_argument('-e', '--epochs', help="size of training set", default=25, type=int)




args = parser.parse_args(["-e80"])

# %%
# --- training params
batch_size=128
patch_size=64
num_epochs = args.epochs
phases = ["train","val"] #how many phases did we create databases for?
validation_phases= ["val"] #when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
                           #additionally, using simply [], will skip validation entirely, drastically speeding things up
    
    
blurparams = {0:[0,0],
              1:[1,3],
              2:[5,7]}
nclasses=len(blurparams)



# %%
#helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# %%
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
 
model = DenseNet(growth_rate=growth_rate, block_config=block_config,
                 num_init_features=num_init_features, 
                 bn_size=bn_size, 
                 drop_rate=drop_rate, 
                 num_classes=nclasses).to(device)

# %%
print(summary(model, (3, patch_size, patch_size)))


# %%
#this defines our dataset class which will be used by the dataloader
class Dataset(object):
    def __init__(self, fname ,img_transform=None):
        #nothing special here, just internalizing the constructor parameters
        self.fname=fname

        self.img_transform=img_transform
        
        with tables.open_file(self.fname,'r') as db:
            self.nitems=db.root.imgs.shape[0]
        
        self.imgs = None
        
    def __getitem__(self, index):
        #opening should be done in __init__ but seems to be
        #an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

        with tables.open_file(self.fname,'r') as db:
            self.imgs=db.root.imgs
            ##apply blur here
            #get the requested image  from the pytable
            label = random.choice([0,1,2])
            img=self.imgs[index,:,:,:]
           
            rand_no=random.randint(blurparams[label][0],blurparams[label][1])
            new_img =  gaussian_filter(img,sigma=(rand_no,rand_no,0)) if rand_no>0 else img
            
        if self.img_transform:
            img_new = self.img_transform(image=new_img)['image']

        return img_new, label, img
    def __len__(self):
        return self.nitems


# %%
    
img_transform = Compose([
        RandomScale(scale_limit=0.1,p=.9),
        PadIfNeeded(min_height=patch_size,min_width=patch_size),        
        VerticalFlip(p=.5),
        HorizontalFlip(p=.5),
        GaussNoise(p=.5, var_limit=(10.0, 50.0)),
        GridDistortion(p=.5, num_steps=5, distort_limit=(-0.3, 0.3), 
                        border_mode=cv2.BORDER_REFLECT),
        ISONoise(p=.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        RandomBrightness(p=.5, limit=(-0.2, 0.2)),
        RandomContrast(p=.5, limit=(-0.2, 0.2)),
        RandomGamma(p=.5, gamma_limit=(80, 120), eps=1e-07),
        MultiplicativeNoise(p=.5, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
        HueSaturationValue(hue_shift_limit=20,sat_shift_limit=10,val_shift_limit=10,p=.9),
        Rotate(p=1, border_mode=cv2.BORDER_REFLECT),
        RandomCrop(patch_size,patch_size),
        ToTensor()
    ])

# %%
dataset={}
dataLoader={}
for phase in phases: #now for each of the phases, we're creating the dataloader
                     #interestingly, given the batch size, i've not seen any improvements from using a num_workers>0
    
    dataset[phase]=Dataset(f"./{dataname}_{phase}.pytable",  img_transform=img_transform)
    dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                shuffle=True, num_workers=12,pin_memory=True) 
    print(f"{phase} dataset size:\t{len(dataset[phase])}")

# %%
#visualize a single example to verify that it is correct
(img, label, img_old)=dataset["train"][0]
fig, ax = plt.subplots(1,2, figsize=(10,4))  # 1 row, 2 columns

#build output showing patch after augmentation and original patch
ax[0].imshow(np.moveaxis(img.numpy(),0,-1))
ax[1].imshow(img_old)

print(label)

# %%
optim = torch.optim.Adam(model.parameters()) #adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
criterion = nn.CrossEntropyLoss() 

# %%
#def trainnetwork():
writer=SummaryWriter() #open the tensorboard visualiser
best_loss_on_test = np.Infinity

%tensorboard --logdir=data/ --host 129.22.136.73 

start_time = time.time()
for epoch in range(num_epochs):
    #zero out epoch based performance variables 
    all_acc = {key: 0 for key in phases} 
    all_loss = {key: torch.zeros(0).to(device) for key in phases} #keep this on GPU for greatly improved performance
    cmatrix = {key: np.zeros((nclasses,nclasses)) for key in phases}

    for phase in phases: #iterate through both training and validation states

        if phase == 'train':
            model.train()  # Set model to training mode
        else: #when in eval mode, we don't want parameters to be updated
            model.eval()   # Set model to evaluate mode

        for ii , (X, label, img_orig) in enumerate(dataLoader[phase]): #for each of the batches
            X = X.to(device)  # [Nbatch, 3, H, W]
            label = label.type('torch.LongTensor').to(device)  # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)

            with torch.set_grad_enabled(phase == 'train'): #dynamically set gradient computation, in case of validation, this isn't needed
                                                            #disabling is good practice and improves inference time

                prediction = model(X)  # [N, Nclass]
                loss = criterion(prediction, label)


                if phase=="train": #in case we're in train mode, need to do back propogation
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss


                all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))

                if phase in validation_phases: #if this phase is part of validation, compute confusion matrix
                    p=prediction.detach().cpu().numpy()
                    cpredflat=np.argmax(p,axis=1).flatten()
                    yflat=label.cpu().numpy().flatten()
                    
                    confusion_matrix = scipy.sparse.coo_matrix( (np.ones(yflat.shape[0], dtype=np.int64), (yflat, cpredflat)),
                            shape=(nclasses, nclasses), dtype=np.int64, ).toarray()

                    cmatrix[phase] = cmatrix[phase] + confusion_matrix  # confusion_matrix(yflat, cpredflat, labels=range(2))
    


        all_acc[phase]=(cmatrix[phase]/cmatrix[phase].sum()).trace()
        all_loss[phase] = all_loss[phase].cpu().numpy().mean()

        #save metrics to tensorboard
        writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
        if phase in validation_phases:
            writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
            for r in range(nclasses):
                for c in range(nclasses): #essentially write out confusion matrix
                    writer.add_scalar(f'{phase}/{r}{c}', cmatrix[phase][r][c],epoch)

    print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                 epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]),end="")    

    #if current loss is the best we've seen, save model state with all variables
    #necessary for recreation
    if all_loss["val"] < best_loss_on_test:
        best_loss_on_test = all_loss["val"]
        print("  **")
        state = {'epoch': epoch + 1,
         'model_dict': model.state_dict(),
         'optim_dict': optim.state_dict(),
         'best_loss_on_test': all_loss,
         'in_channels': in_channels,
         'growth_rate':growth_rate,
         'block_config':block_config,
         'num_init_features':num_init_features,
         'bn_size':bn_size,
         'drop_rate':drop_rate,
         'nclasses':nclasses}


        torch.save(state, f"{dataname}_densenet_best_model.pth")
    else:
        print("")
