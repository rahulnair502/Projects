# %%
# coding: utf-8
tables.file._open_files.close_all()

# %%
#Created by Rahul Nair
dataname="blurry_classification"

patch_size=64 #size of the tiles to extract and save in the database, must be >= to training size
stride_size= 16 #distance to skip between patches, 1 indicated pixel wise extraction, patch_size would result in non-overlapping tiles
mirror_pad_size=16 # number of pixels to pad *after* resize to image with by mirroring (edge's of patches tend not to be analyzed well, so padding allows them to appear more centered in the patch)
test_set_size=.1 # what percentage of the dataset should be used as a held out validation/testing set
phases = ["train","val"] #how many phases did we create databases for?
validation_phases= ["val"] #when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
                           #additionally, using simply [], will skip validation entirely, drastically speeding things up

sample_level=0


#-----Note---
#One should likely make sure that  (nrow+mirror_pad_size) mod patch_size == 0, where nrow is the number of rows after resizing
#so that no pixels are lost (any remainer is ignored)


# %%
import tables
import numpy as np
import argparse
import PIL
import sklearn.feature_extraction.image
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from  skimage.color import rgb2gray
import cv2
import os
import glob
#openslide doesnt always import correctly for windows so this is neccessary if that is the case
#os.add_dll_directory(r"C:\Users\Rahul Nair\anaconda3\Lib\site-packages\openslide\openslide-win64-20171122\bin")
os.environ['PATH'] = 'C:\\research\\openslide\\bin' + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
import openslide
import random
import os,sys
from sklearn import model_selection
from WSI_handling import wsi


# %%
seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the cross validation setup
random.seed(seed) # set the seed
print(f"random seed (note down for reproducibility): {seed}")


# %%
#this is the level the WSI will be read, keep this level constant in output generation
def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 


# %%
def random_subset(a, b, nitems):
    assert len(a) == len(b)
    idx = np.random.randint(0,len(a),nitems)
    return a[idx], b[idx]



# %%
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just incase we need it later,
labelAtom = tables.StringAtom(itemsize=255)


# %%
block_shape=np.array((patch_size,patch_size,3)) #block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
filters=tables.Filters(complevel=6, complib='zlib') #we can also specify filters, such as compression, to improve storage speed

# %%
parser = argparse.ArgumentParser(description='Extract tiles for analysis')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.svs, or tsv file containing list of files to analyze",
                    nargs="*")
parser.add_argument('-m', '--enablemask', action="store_true")
parser.add_argument('-t', '--trainsize', help="size of training set", default=10000, type=int)
parser.add_argument('-v', '--valsize', help="size of validation set", default=2000, type=int)
parser.add_argument('-p', '--basepath',
                        help="base path to add to file names, helps when producing data using existing output file as input",
                        default="", type=str)


args = parser.parse_args([r"/mnt/data/home/rxn198/testing_2/12.svs"])

max_number_samples={"train":args.trainsize,"val":args.valsize}     #---- RAHUL: these should be command line parameters
# %%
files = []
basepath = args.basepath  #
basepath = basepath + os.sep if len(
    basepath) > 0 else ""  # if the user supplied a different basepath, make sure it ends with an os.sep
if len(args.input_pattern) > 1:  # bash has sent us a list of files
    files = args.input_pattern
elif args.input_pattern[0].endswith("tsv"):  # user sent us an input file
    # load first column here and store into files
    with open(args.input_pattern[0], 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            files.append(basepath + line.strip().split("\t")[0])
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.basepath + args.input_pattern[0])



# %%
storage={} #holder for future pytables
for phase in phases: #now for each of the phases, we'll loop through the files
    storage[phase]={}
    hdf5_file = tables.open_file(f"./{dataname}_{phase}.pytable", mode='w') #open the respective pytable
    storage[phase]["filenames"] = hdf5_file.create_earray(hdf5_file.root, 'filenames', filenameAtom, (0,)) #create the array for storage
    
    storage[phase]["imgs"]= hdf5_file.create_earray(hdf5_file.root, "imgs", img_dtype,  
                                              shape=np.append([0],block_shape), 
                                              chunkshape=np.append([1],block_shape),
                                             filters=filters)


# %%
for filei in tqdm(files): #now for each of the files
    osh  = openslide.OpenSlide(filei)
    osh_mask  = wsi(filei)
    mask_level_tuple = osh_mask.get_layer_for_mpp(8)
    mask_level = mask_level_tuple[0]

    
    if(args.enablemask):
        mask=cv2.imread(os.path.splitext(filei)[0]+'.png') #--- assume mask has png ending in same directory 
    else:
        
        img = osh.read_region((0, 0), mask_level, osh.level_dimensions[mask_level])
        img = np.asarray(img)[:, :, 0:3]
        imgg=rgb2gray(img)
        mask=np.bitwise_and(imgg>0 ,imgg <230/255)
        kernel = np.ones((5,5), np.uint8)
        mask = np.float32(mask)
        mask =  cv2.erode(mask, kernel, iterations=4)
       


    [rs,cs]=mask.nonzero()

    for phase in phases:
        [prs,pcs]=random_subset(rs,cs,min(max_number_samples[phase],len(rs)))

        #RAHUL, add a check here to make sure that the mask has the same aspect ratio as the osh
        downsampled=int(osh.level_dimensions[sample_level][0]/mask.shape[1])

        for i, (r,c) in tqdm(enumerate(zip(prs,pcs)),total =len(prs), desc=f"innter2-{phase}", leave=False):

            io = np.asarray(osh.read_region((c*downsampled-patch_size//2, r*downsampled-patch_size//2), sample_level, (patch_size, patch_size)))
            img = np.asarray(io)[:, :, 0:3]
            io = io[:, :, 0:3]  # remove alpha channel
           
            
            imgg=rgb2gray(img)
            mask2=np.bitwise_and(imgg>0 ,imgg <230/255)
            plt.imshow(mask2,cmap="gray")
            
           
           
            if np.count_nonzero(mask2 == True) > (patch_size * patch_size) /8:
                 storage[phase]["imgs"].append(io[None,::])
                 storage[phase]["filenames"].append([f'{filei}_{r}_{c}']) #add the filename to the storage array
           
    osh.close()
tables.file._open_files.close_all()


# %%
