# -*- coding: utf-8 -*-
"""
Intaretinal cyst fluid detection: Getting to know the data deeper
    This scripts explores how the raw data was given, for this we used
    Spider to do a data inspection in the numpy arrays.
    
@author: Gabriel Raya.
"""

#####################################################
# Load Libraries
####################################################  
#from tqdm import tnrange, tqdm_notebook
import os
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
#from PIL import Image
matplotlib.rcParams['figure.figsize'] = [8, 6]
#from IPython.display import clear_output

#%matplotlib inline

#import tensorflow as tf
from keras.models import Model#, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization # Dropout, Cropping2D, Reshape,
#from keras import optimizers
from keras.optimizers import SGD#, Adam
#from keras import regularizers
import keras.callbacks
#from utils import *
#from keras import backend as K
#####################################################
# Helper functions
####################################################  

def get_file_list(path, ext=''):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])

def load_img(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))



class Image:
  
    def __init__(self, img, label=None, name=None):
        '''
        Inputs:
        img: image as a numpy array
        label: labels of the image as np image
        name: the filename of the image
        '''
        self.img = img
        self.label = label
        self.name = name
    
    def get_img(self):
        return self.img
  
    def get_label(self):
        return self.label
  
    def get_lenght(self):
        return len(self.imgs)
  
    def show_image(self):
        f, axes = plt.subplots(1, 2)
        for ax, im, t in zip(axes, 
                           (self.img, self.lbl), 
                           ('RGB image', 
                            'Manual annotation; Range: [{}, {}]'.format(self.lbl.min(), 
                                                                        self.lbl.max()))):
            ax.imshow(im)
            ax.set_title(t)
        plt.show()
      

# TODO: add name to dataset
class DataSet:
    
    def __init__(self, imgs, lbls=None):
        self.imgs = imgs
        self.lbls = lbls
    
    def get_lenght(self):
        return len(self.imgs)
    
    def show_image(self, i):
        if self.lbls != None:
            f, axes = plt.subplots(1, 2)
            for ax, im, t in zip(axes, 
                                 (self.imgs[i], self.lbls[i]), 
                                 ('RGB image', 
                                  'Manual annotation; Range: [{}, {}]'.format(self.lbls[i].min(), 
                                                                              self.lbls[i].max()))):
                ax.imshow(im)
                ax.set_title(t)
        else:
            plt.imshow(self.imgs[i])
            plt.title('RGB image')
        plt.show()
        
        
class BatchCreator:
    
    def __init__(self, dataset, target_size):
        #self.patch_extractor = patch_extractor
        self.target_size = target_size # size of the output, can be useful when valid convolutions are used
        
        self.imgs = [image.img for image in dataset]
        self.lbls = [image.label for image in dataset]
                
        self.n = len(self.imgs)
        #self.patch_size = self.patch_extractor.patch_size
    
    def create_image_batch(self, batch_size):
        '''
        returns a single augmented image (x) with corresponding labels (y) in one-hot structure
        '''
        x_data = np.zeros((batch_size, *self.target_size, 1))
        y_data = np.zeros((batch_size, *self.target_size, 2)) # one-hot encoding with 2 classes
        
        for i in range(0, batch_size):
        
            random_index = np.random.choice(len(self.imgs)) # pick random image
            img, lbl = self.imgs[random_index], self.lbls[random_index] # get image and segmentation map
            
           # clear_output()
            #patch_img, patch_lbl = self.patch_extractor.get_patch(img, lbl) # when image size is equal to patch size, this line is useless...
          #  img, lbl = 
            # crop labels based on target_size
            #h, w, _ = patch_lbl.shape
            #ph = (self.patch_extractor.patch_size[0] - self.target_size[0]) // 2
            #pw = (self.patch_extractor.patch_size[1] - self.target_size[1]) // 2
            #patch_img = patch_img.reshape(tuple(list(patch_img.shape) + [1]))
            x_data[i, :, :, :] = img
            y_data[i, :, :, 0] = 1 - lbl
            y_data[i, :, :, 1] = lbl
        
        return (x_data.astype(np.float32), y_data.astype(np.float32))
    
    def get_image_generator(self, batch_size):
        '''returns a generator that will yield image-batches infinitely'''
        while True:
            yield self.create_image_batch(batch_size)
            
def pad_prediction(prediction, input_size, pad_with=-1.0):
    """Only for visualization purpose, it introduces artificial -1."""
    pad_pred = pad_with * np.ones(input_size).astype(float)
    pred_size = prediction.shape
    D = ((input_size[0]-pred_size[0])//2, (input_size[1]-pred_size[1])//2)
    pad_pred[D[0]:D[0]+pred_size[0], D[1]:D[1]+pred_size[1]] = prediction
    return pad_pred

def binarize_dataset(dataset): 
    return DataSet(dataset.imgs, [(x>0).astype(int) for x in dataset.lbls])

def apply_model(model, dataset, experiment_name='basic_unet', make_submission_file=False):
    """Apply a given model to the test set, optionally makes a submission file in ZIP format."""
    
    output_dir = os.path.join('.', experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    imgs = [image.img for image in dataset]

    for i in range(len(imgs)):
        fig = plt.figure(figsize=(10,20))
        input_img = np.expand_dims(imgs[i], axis=0)/255.
        output = model.predict(input_img, batch_size=1)[0, :, :]
        plt.subplot(1, 2, 1); plt.imshow(imgs[i].reshape(480,480))
        plt.subplot(1, 2, 2); plt.imshow(np.argmax(output, axis=2))
        if make_submission_file:
            prediction = Image.fromarray(np.argmax(output, axis=2).astype(np.uint8))
            prediction.save(os.path.join(output_dir, '{}.png'.format(i)))
        plt.show()
        
    if make_submission_file:
        shutil.make_archive('results', 'zip', output_dir)


           

def crop(masks, lost_border):
    ph = lost_border[0] // 2
    pw = lost_border[1] // 2
    h, w = masks[0].shape    
    return np.array(masks)[:, ph:h-ph, pw:w-pw]   

def calculate_dice(x, y):
    '''returns the dice similarity score, between two boolean arrays'''
    return 2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y))
    
class Logger(keras.callbacks.Callback):

    def __init__(self, validation_data, lost_border, data_dir, model_name):
        self.val_imgs = np.array([image.img for image in validation_data]) / 255.
        self.val_lbls = crop([image.label for image in validation_data], lost_border)
        self.model_filename = os.path.join(data_dir, model_name + '.h5')
        
        self.losses = []
        self.dices = []
        self.best_dice = 0
        self.best_model = None
        
        self.predictions = None
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.losses.append(logs.get('acc'))
    
    def on_epoch_end(self, batch, logs={}):
        dice = self.validate()
        self.dices.append([len(self.losses), dice])
        if dice > self.best_dice:
            self.best_dice = dice
            self.model.save(self.model_filename) # save best model to disk
            print('best model saved as {}'.format(self.model_filename))
        self.plot()   
    
    def validate(self):
        self.predictions = self.model.predict(self.val_imgs, batch_size=1)
        predicted_lbls = np.argmax(self.model.predict(self.val_imgs, batch_size=1), axis=3)
        x = self.val_lbls>0
        y = predicted_lbls>0
        return calculate_dice(x, y)

    def plot(self):
        #clear_output()
        N = len(self.losses)
        plt.figure(figsize=(50, 10))
        plt.subplot(1, 5, 1)
        plt.plot(range(0, N), self.losses); plt.title('losses')
        plt.subplot(1, 5, 2)
        plt.plot(*np.array(self.dices).T); plt.title('dice')
        plt.subplot(1, 5, 3)
        plt.imshow(self.val_imgs[1].reshape(480,480)); plt.title('RGB image')
        plt.subplot(1, 5, 4)
        plt.imshow(np.argmax(self.predictions[1], axis=2)); plt.title('prediction')
        plt.subplot(1, 5, 5)
        plt.imshow(self.val_lbls[1]); plt.title('ground truth')
        plt.show()
        
# function to train a model
def train_model(model, training_params):
    patch_size = training_params['patch_size']
    target_size = training_params['target_size']
    batch_size = training_params['batch_size']
    loss = training_params['loss']
    metrics = training_params['metrics']
    logger = training_params['logger']
    epochs = training_params['epochs']
    steps_per_epoch = training_params['steps_per_epoch']
    optimizer = training_params['optimizer']
    training_dataset = training_params['training_dataset']
    validation_dataset = training_params['validation_dataset']
        
    # batch generator 
    #patch_generator = PatchExtractor(patch_size)
    batch_generator = BatchCreator(training_dataset, target_size=target_size)
    image_generator = batch_generator.get_image_generator(batch_size)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # train the model
    model.fit_generator(generator=image_generator, 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs,
                        callbacks=[logger])
    
    
def pad_n_crop_(img, cropx, cropy):
    img = np.pad(img, ((0,0),(0,0),(0,30)), mode='edge')

    _, y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:, starty:starty+cropy,startx:startx+cropx]

def show_image(img, lbl):
    matplotlib.rcParams['figure.figsize'] = (10, 6)
    s, h, w = img.shape 
    for i in range(s):
        plt.subplot(1,2,1)
        plt.imshow(img[i])
        plt.title('RGB image')

        plt.subplot(1,2,2)
        plt.imshow(lbl[i])
        plt.title('Manual annotation')
        plt.show()
#####################################################
# Set data directory
####################################################  
DATA_FOLDER = './cyst_segmentation_ISMI_training_set/'                                  # Local System
#DATA_FOLDER = 'gdrive/Team Drives/ISMI-FinalProject/cyst_segmentation_ISMI_training_set/' # Colab

#####################################################
# Data inspection for 1st images
#################################################### 

# raw data in ITK format
raw_img_filename = os.path.join(DATA_FOLDER,'images/pat001_im001.mhd')
out_img_filename = os.path.join(DATA_FOLDER,'annotations/pat001_im001.mhd')

# read ITK files using SimpleITK
raw_img = sitk.ReadImage(raw_img_filename)
out_img = sitk.ReadImage(out_img_filename)

# print image information
print('image size: {}'.format(raw_img.GetSize()))
print('image origin: {}'.format(raw_img.GetOrigin()))
print('image spacing: {}'.format(raw_img.GetSpacing()))
print('image width: {}'.format(raw_img.GetWidth()))
print('image height: {}'.format(raw_img.GetHeight()))
print('image depth: {}'.format(raw_img.GetDepth()))

# convert the ITK image into numpy format
out_np = sitk.GetArrayFromImage(out_img)
raw_np = sitk.GetArrayFromImage(raw_img)

# visualize the two numpy arrays (first b-scans)
plt.subplot(1,2,1)
plt.imshow(raw_np[11], cmap='gray')
plt.title('raw data')
plt.subplot(1,2,2)
plt.imshow(out_np[11], cmap='gray')
plt.title('gray-level data (target)')
plt.show()
#plt.savefig('./images/img1_raw_data.png')

# print max and min values (0-255, 8-bit image)
print('image max: {}'.format(np.amax(raw_np[11])))
print('image min: {}'.format(np.amin(raw_np[11])))

print('image max: {}'.format(np.amax(out_np[11])))
print('image min: {}'.format(np.amin(out_np[11])))
#####################################################
# Data inspection for the 122 volumes
#################################################### 


###############################################################################
###############################################################################
# First U-net
###############################################################################
###############################################################################

###############################################################################
# Define datasets
############################################################################### 

DATA_FOLDER = './cyst_segmentation_ISMI_training_set/'   

# get path names list of raw data in ITK format
x_img_files = get_file_list(os.path.join(DATA_FOLDER,'images/'), 'mhd')
y_img_files = get_file_list(os.path.join(DATA_FOLDER,'annotations/'), 'mhd')

# load images + manual annotations from the training set and store them in lists.
# 10 is used to reduce computational work
train_imgs = [load_img(x_img_files[f]) for f in range(10)]
train_lbls = [load_img(y_img_files[f]) for f in range(10)]

def plot_image_no_axis(img, save_path):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(img) 
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(save_path, bbox_inches=extent)

plot_image_no_axis(train_imgs[1][17],'./images/pat1_scan18.png')
plot_image_no_axis(train_lbls[1][17],'./images/pat1_scan18_anno.png')
plot_image_no_axis(train_imgs[2][17],'./images/pat2_scan17.png')
plot_image_no_axis(train_lbls[2][17],'./images/pat2_scan17_anno.png')
plot_image_no_axis(train_imgs[3][16],'./images/pat3_scan16.png')
plot_image_no_axis(train_lbls[3][16],'./images/pat3_scan16_anno.png')
plot_image_no_axis(train_imgs[4][16],'./images/pat4_scan16.png')
plot_image_no_axis(train_lbls[4][16],'./images/pat4_scan16_anno.png')


# shuffle order of training images and manual annotations
indexes = list(range(len(train_imgs)))
random.shuffle(indexes)
train_imgs = list(np.asarray(train_imgs)[indexes])
train_lbls = list(np.asarray(train_lbls)[indexes])

# sanity check
# Display patient 6 b-scans
i = 1
show_image(train_imgs[i],train_lbls[i])

###########Split
validation_percent = 0.2 
n_validation_imgs = int(validation_percent * len(train_imgs))


# use the first images as validation
validation_dataset = DataSet(train_imgs[:n_validation_imgs], train_lbls[:n_validation_imgs])

# the rest as training
training_dataset = DataSet(train_imgs[n_validation_imgs:], train_lbls[n_validation_imgs:])

n_tra_imgs = training_dataset.get_lenght()
n_val_imgs = validation_dataset.get_lenght()

print('{} training images'.format(n_tra_imgs))
print('{} validation images'.format(n_val_imgs))

train_data = []
volumen_img = training_dataset.imgs[0] # get i volume
volumen_lbl = training_dataset.lbls[0]
volumen_img_cropped = pad_n_crop_(volumen_img, 480, 480).reshape(volumen_img.shape[0],480,480,1)
volumen_lbl_cropped = pad_n_crop_(volumen_lbl, 480, 480)
b_scans = volumen_img.shape[0]
# iterates over b-scans
for i in range(b_scans):
    b_scan = Image(volumen_img_cropped[i,:,:], volumen_lbl_cropped[i,:,:])
    train_data.append(b_scan)
            
            
# get scans helper function
def get_scans(training_dataset, train_data):
    for volumen in range(training_dataset.get_lenght()):
        volumen_img = training_dataset.imgs[volumen] # get i volume
        volumen_lbl = training_dataset.lbls[volumen]
        volumen_img_cropped = pad_n_crop_(volumen_img, 480, 480).reshape(volumen_img.shape[0],480,480,1)
        volumen_lbl_cropped = pad_n_crop_(volumen_lbl, 480, 480)
        b_scans = volumen_img.shape[0]
        # iterates over b-scans
        for i in range(b_scans):
            b_scan = Image(volumen_img_cropped[i,:,:], volumen_lbl_cropped[i,:,:])
            train_data.append(b_scan)
            
#get b-scans
train_data = []
get_scans(training_dataset, train_data)
validation_data = []
get_scans(validation_dataset, validation_data)

i = 11
show_image(train_data[i].img.reshape(1,480,480),train_data[i].label.reshape(1,480,480))
      

n_tra_imgs = len(train_data)
n_val_imgs = len(validation_data)

print('{} training b-scans images'.format(n_tra_imgs))
print('{} validation b-scans images'.format(n_val_imgs))

        
# test converstion> 1 is added because the show_image function recieves volumetric
for i in range(30):
    show_image(train_data[i].img.reshape(1,480,480),train_data[i].label.reshape(1,480,480))

train_data[10].img.shape



# define parameters for patch generator and batch creator
#patch_size = (32, 32) # input size
target_size = (480, 480) # output size, might be the same as input size, but might be smaller, if valid convolutions are used
batch_size = 16 # number of patches in a mini-batch

# intialize patch generator and batch creator
#patch_generator = PatchExtractor(patch_size)
batch_generator = BatchCreator(train_data, target_size=target_size)

# get one minibatch
x_data, y_data = batch_generator.create_image_batch(batch_size)

#test
#i = 5
#show_image(x_data[i].reshape(1,480,480),y_data[i].reshape(1,480,480))


print(x_data.shape)
print(y_data.shape)
print(y_data[0, :, :, 0].squeeze().shape)
for i in range(batch_size):
    plt.subplot(1, 3, 1)
    plt.imshow(x_data[i].reshape(480,480)); plt.title('RGB image')
    plt.subplot(1, 3, 2)
    plt.imshow(pad_prediction(y_data[i, :, :, 0].squeeze(), (480,480))); plt.title('Label map class 0')
    plt.subplot(1, 3, 3)
    plt.imshow(pad_prediction(y_data[i, :, :, 1].squeeze(), (480,480))); plt.title('Label map class 1')
    plt.show()
    
        
    
# Create a function that builds a U-Net block, containing conv->(batchnorm->)conv->(batchnorm),
# where batchnorm is optional and can be selected via input parameter.
# The function returns the output of a convolutional (or batchnorm) layer "cl"
def unet_block(inputs, n_filters, batchnorm=False, name= None):
    
    # >> YOUR CODE HERE <<
    cl = Conv2D(n_filters,3, activation = 'relu', padding = 'same', name = name+"1" )(inputs)
    cl = BatchNormalization()(cl) if batchnorm else cl
    cl = Conv2D(n_filters,3,activation = 'relu', padding = 'same', name = name+"2")(cl)
    cl = BatchNormalization()(cl) if batchnorm else cl
    
    return cl

def build_unet_1(printmodel=False):
    
    inputs = Input(shape=(480, 480, 1))

    # First conv pool
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    # Second conv pool
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    # Third conv pool
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    # Fourth conv pool
    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)

    # First up-conv
    u2 = UpSampling2D()(c4)
    m2 = concatenate([c3, u2])
    cm2 = Conv2D(64, 3, activation='relu', padding='same')(m2)
    cm2 = Conv2D(64, 3, activation='relu', padding='same')(cm2)

    # Second up-conv
    u3 = UpSampling2D()(cm2)
    m3 = concatenate([c2, u3])
    cm3 = Conv2D(32, 3, activation='relu', padding='same')(m3)
    cm3 = Conv2D(32, 3, activation='relu', padding='same')(cm3)

    # Third up-conv
    u4 = UpSampling2D()(cm3)
    m4 = concatenate([c1, u4])
    cm4 = Conv2D(16, 3, activation='relu', padding='same')(m4)
    cm4 = Conv2D(16, 3, activation='relu', padding='same')(cm4)

    # Output
    predictions = Conv2D(2, 1, activation='softmax')(cm4)

    model = Model(inputs, predictions)
    
    if printmodel:
        print(model.summary())
    
    return model


def build_unet_2(initial_filters=16, n_classes=2, batchnorm=True, printmodel=True):

    # build U-Net again using unet_block function
    inputs = Input(shape=(480, 480, 1))

    # CONTRACTION PART

    # First conv pool
    c1 = unet_block(inputs, initial_filters, batchnorm,name ='first')
    p1 = MaxPooling2D()(c1)

    # Second conv pool
    c2 = unet_block(p1, 2*initial_filters, batchnorm,name='second')
    p2 = MaxPooling2D()(c2)

    # Third conv pool
    c3 = unet_block(p2, 4*initial_filters, batchnorm,name='third')
    p3 = MaxPooling2D()(c3)

    # Fourth conv
    c4 = unet_block(p3, 8*initial_filters, batchnorm, name='fourth')

    # EXPANSION PART

    # First up-conv
    u2 = UpSampling2D()(c4)
    m2 = concatenate([c3, u2])
    cm2 = unet_block(m2, 4*initial_filters, batchnorm, name ='fifth')

    # Second up-conv
    u3 = UpSampling2D()(cm2)
    m3 = concatenate([c2, u3])
    cm3 = unet_block(m3, 2*initial_filters, batchnorm, name ='sixth')

    # Third up-conv
    u4 = UpSampling2D()(cm3)
    m4 = concatenate([c1, u4])
    cm4 = unet_block(m4, initial_filters, batchnorm, name = 'seventh')

    # Output
    predictions = Conv2D(n_classes, 1, activation='softmax', name = 'last')(cm4)

    model = Model(inputs, predictions)
    
    if printmodel:
        print(model.summary())
    
    return model


#############
unet_1 = build_unet_2()

unet_1 = build_unet_1()

apply_model(unet_1, validation_data)
# apply the model to the validation set
data_dir = '.'
#apply_model(unet_1, validation_data)

# training parameters

model_name = 'unet_1'

training_params = {}
training_params['learning_rate'] = 0.001
training_params['patch_size'] = (480, 480) # input size
training_params['target_size'] = (480, 480) # output size, might be the same as input size, but might be smaller, if valid convolutions are used
training_params['batch_size'] = 16 # number of patches in a mini-batch
training_params['steps_per_epoch'] = 25 # number of iterations per epoch
training_params['epochs'] = 1 # number of epochs

training_params['optimizer'] = SGD(lr=training_params['learning_rate'], momentum=0.9, nesterov=True)

training_params['loss'] = ['categorical_crossentropy']
training_params['metrics'] = ['accuracy']
training_params['training_dataset'] = train_data
training_params['validation_dataset'] = validation_data

# initialize a logger, to keep track of information during training
lost_border = ((training_params['patch_size'][0]-training_params['target_size'][0])//2, (training_params['patch_size'][1]-training_params['target_size'][1])//2)
training_params['logger'] = Logger(validation_data, lost_border, data_dir, model_name)

# train model
train_model(unet_1, training_params)

apply_model(unet_1,validation_data)



loss_function = 'categorical_crossentropy'
optimizer =SGD(lr=0.0001, momentum=0.9, nesterov=True)

unet_1.compile(optimizer, loss_function)

unet_1.fit_generator(batch_generator.get_image_generator(32), steps_per_epoch = 50, epochs = 5)




