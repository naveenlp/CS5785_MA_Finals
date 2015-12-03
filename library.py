
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import json
from sklearn.decomposition import PCA

# READ IN DATA AND STORE IN VARIABLES

def fp(filename):
    return "../../CS5785-final-data/"+filename

# training data
train_cnn = np.load(fp("alexnet_feat_train.npy"))
train_bow = np.load(fp("SIFTBoW_train.npy"))
train_labels = np.genfromtxt(fp("train.txt"),dtype=None)
train_filenames = train_labels[:,0]
train_labels = train_labels[:,1]
train_attribs = np.array([map(float, line.split(',')) for line in np.genfromtxt(fp("attributes_train.txt"),dtype=None)[:,1]])

# testing data
test_cnn = np.load(fp("alexnet_feat_test.npy"))
test_bow = np.load(fp("SIFTBoW_test.npy"))
test_filenames = np.genfromtxt(fp("test.txt"),dtype=None)
test_attribs = np.array([map(float, line.split(',')) for line in np.genfromtxt(fp("attributes_test.txt"),dtype=None)[:,1]])

# 10k data
tenk_cnn = np.load(fp("alexnet_feat_10k.npy"))
tenk_bow = np.load(fp("SIFTBoW_10k.npy"))
tenk_filenames = np.genfromtxt(fp("10k_list.txt"),dtype=None)
with open(fp('captions.json')) as data_file:    
        captions_json = json.load(data_file)
        
# metadata
attribute_values = [line.rstrip().replace("'", "") for line in open(fp('attributes_list.txt'))]


# In[12]:

# HELPER METHODS
def get_filename(index, datatype = "train"):
    if(datatype == "train"):
        name = train_filenames[index]
    elif (datatype == "test"):
        name = test_filenames[index]
    else:
        name = tenk_filenames[index]
    return name

def display_image(filename, datatype = "train"):
    if((datatype == "train") or (datatype == "test")):
        path = fp('images/') + datatype + '/' + filename
    else:
        path = fp('10k_images/') + filename
        
    im = Image.open(path)
    size = 200,180
    imt = im.convert('RGB')
    imt.thumbnail(size, Image.ANTIALIAS)
    return imt

def display_image2(filename, datatype = "train"):
    if((datatype == "train") or (datatype == "test")):
        path = fp('images/') + datatype + '/' + filename
    else:
        path = fp('10k_images/') + filename
    
    im = Image.open(path).convert('RGB')
    arr = np.asarray(im)
    plt.imshow(arr, interpolation='nearest')
    plt.show()

# sample call
# display_image(get_filename(8, 'att'), 'att')

def kaggle_output(outputs):
    kaggle_output = np.vstack([test_filenames,outputs]).transpose()
    print kaggle_output
    np.savetxt("../kaggle_output.csv", kaggle_output, delimiter=",", header="ID,Category", comments='', fmt="%s")
    

def get_pca(data, count = 50):
    pca = PCA(n_components = count)
    return pca.fit_transform(data)


# In[33]:

get_ipython().system(u'ipython nbconvert --to=python library.ipynb')

