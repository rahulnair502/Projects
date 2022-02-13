#Created by Rahul Nair
import numpy as np
import matplotlib.pyplot as plt
import argparse

from WSI_handling import wsi
import sklearn.feature_extraction.image
import matplotlib.pyplot as plt

import matplotlib.cm
import cv2
import torch

from torchvision.models import DenseNet

from tqdm.autonotebook import tqdm

from  skimage.color import rgb2gray
import os
from os import path
#os.add_dll_directory(r"C:\Users\Rahul Nair\anaconda3\Lib\site-packages\openslide\openslide-win64-20171122\bin")
import openslide
from skimage.external.tifffile import TiffWriter
import openslide
import ttach as tta
#command line arguments for easy running
parser = argparse.ArgumentParser(description='Make output for entire image using DenseNet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-r', '--resize', help="resize factor 1=1x, 2=2x, .5 = .5x", default=1, type=float)
parser.add_argument('-m', '--model', help="model", default="best_model.pth", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)
parser.add_argument('-v', '--mask_val', help="mask_val variable from first script", default=230, type=int)
parser.add_argument('-t', '--enablemask', action="store_true")
parser.add_argument('-k', '--mask_level',help="level of input mask", default=2, type=int)
                    
##specify location of model in -m, and folder with images in -
args = parser.parse_args(["*.svs","-p64","-mblurry_classification_densenet_best_model.pth","-b/mnt/data/home/rxn198/brTest_1.svs","-v240"])
# +

device = torch.device(args.gpuid if args.gpuid!=-2 and torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                 num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                 drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["nclasses"]).to(device)



#model = tta.ClassificationTTAWrapper(pre_model, tta.aliases.five_crop_transform(65,35))
#the model is loaded
model.load_state_dict(checkpoint["model_dict"])
model.eval()

print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# -
#files are collected
import glob
images = glob.glob(args.basepath) 
print(images)
for slide in images:
  print(slide)
  fname=slide
  osh_mask  = wsi(fname)
  mask_level_tuple = osh_mask.get_layer_for_mpp(8)
  mask_level = mask_level_tuple[0]
  img = osh_mask.read_region((0, 0), mask_level, osh_mask["img_dims"][mask_level])


  def divide_batch(l, n): 
      for i in range(0, l.shape[0], n):  
          yield l[i:i + n,::] 
            
   
  if(args.enablemask):
    mask=cv2.imread(os.path.splitext(slide)[0]+'.png') #--- assume mask has png ending in same directory 
    width = int(img.shape[1] )
    height = int(img.shape[0])
    dim = (width, height)
    mask = cv2.resize(mask,dim)
    mask = np.float32(mask)
    
    
    
  else:
    #add mask creation which skips parts of image
        
        
        img = np.asarray(img)[:, :, 0:3]
        
        imgg=rgb2gray(img)
        mask_value = args.mask_val
        mask=np.bitwise_and(imgg>0 ,imgg <230/255)
        kernel = np.ones((5,5), np.uint8)
        mask = np.float32(mask)
        width = int(mask.shape[1] )
        height = int(mask.shape[0])
        dim = (width, height)
        mask =  cv2.erode(mask, kernel, iterations=2)         


  
   
  plt.imshow(mask)
  plt.show()  

  cmap= matplotlib.cm.tab10

  #ensure that this level is the same as the level used in training
  osh  = openslide.OpenSlide(fname)
  osh.level_dimensions
  mask_level_tuple_2 = osh_mask.get_layer_for_mpp(1)
  #level = mask_level_tuple_2[0]
  level = 0
  ds=int(osh.level_downsamples[level])

  patch_size = args.patchsize
  #change the stride size to speed up the process, must equal factor between levels 
  stride_size = patch_size//4
  tile_size=stride_size*8*2
  tile_pad=patch_size-stride_size
  nclasses=3
  batch_size = args.batchsize 


  shape=osh.level_dimensions[level]
  shaperound=[((d//tile_size)+1)*tile_size for d in shape]

  #patches are extracted and analyzed
  npmm=np.zeros((shaperound[1]//stride_size,shaperound[0]//stride_size,3),dtype=np.uint8)
  for y in tqdm(range(0,osh.level_dimensions[0][1],round(tile_size * osh.level_downsamples[level])), desc="outer"):
      for x in tqdm(range(0,osh.level_dimensions[0][0],round(tile_size * osh.level_downsamples[level])), desc=f"innter {y}", leave=False):

          #if skip
        
        
        
          maskx=int(x//osh.level_downsamples[mask_level])
          masky=int(y//osh.level_downsamples[mask_level])
        
        
          if((np.any(maskx>= mask.shape[1])) or np.any(masky>= mask.shape[0]) or not np.any(mask[masky,maskx])): #need to handle rounding error 
              continue
        
        
          output = np.zeros((0,nclasses,patch_size//patch_size,patch_size//patch_size))
          io = np.asarray(osh.read_region((x, y), level, (tile_size+tile_pad,tile_size+tile_pad)))[:,:,0:3] #trim alpha
        
          arr_out=sklearn.feature_extraction.image.extract_patches(io,(patch_size,patch_size,3),stride_size)
          arr_out_shape = arr_out.shape
          arr_out = arr_out.reshape(-1,patch_size,patch_size,3)
        
            #patches are ran through model
        
          for batch_arr in divide_batch(arr_out,batch_size):
        
              arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

              # ---- get results
              output_batch = model(arr_out_gpu)

             # --- pull from GPU and append to rest of output 
              output_batch = output_batch.detach().cpu().numpy()

              output_batch_color=cmap(output_batch.argmax(axis=1), alpha=None)[:,0:3]
              output = np.append(output,output_batch_color[:,:,None,None],axis=0)
             
        
          output = output.transpose((0, 2, 3, 1))
         
          #turn from a single list into a matrix of tiles
          output = output.reshape(arr_out_shape[0],arr_out_shape[1],patch_size//patch_size,patch_size//patch_size,output.shape[3])
          output3 = output
        
          #turn all the tiles into an image
          output=np.concatenate(np.concatenate(output,1),1)
          output4 = output
          
          
          npmm[y//stride_size//ds:y//stride_size//ds+tile_size//stride_size,x//stride_size//ds:x//stride_size//ds+tile_size//stride_size,:]=output*255 #need to save uint8

  #change the color to red green and blue. red signifies high blur, green signifies no blur, blue signifies medium blur
  data = npmm
  data1 = data
  data2 = data
  data3 = data
  r1, g1, b1 = 255, 127 ,14  # Original value
  r2, g2, b2 = 0, 0, 255 # Value that we want to replace it with


  red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
  mask = (red == r1) & (green == g1) & (blue == b1)
  data1[:,:,:3][mask] = [r2, g2, b2]

  r1, g1, b1 = 44, 160 ,44  # Original value
  r2, g2, b2 = 255, 0, 0 # Value that we want to replace it with


  red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
  mask = (red == r1) & (green == g1) & (blue == b1)
  data2[:,:,:3][mask] = [r2, g2, b2]

  r1, g1, b1 = 31, 119 ,180  # Original value
  r2, g2, b2 = 0, 255, 0 # Value that we want to replace it with


  red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
  mask = (red == r1) & (green == g1) & (blue == b1)
  data3[:,:,:3][mask] = [r2, g2, b2]
  final_output = data1+data2+data3
  
  with TiffWriter('output_'+slide[slide.rfind('/') + 1:slide.rfind('.')-1 ] + '.tif', bigtiff=True, imagej=True) as tif:
  
       tif.save(final_output, compress=6, tile=(16,16) ) 
     
# -
