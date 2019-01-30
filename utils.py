import os
import numpy as np
import copy
from celebA_estimators import *
from CelebAGenerator import *
import tensorflow as tf
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import scipy.misc as scipy
	
#fourier model
def loading_images(Orig_Path,random_restarts):
    X_Orig = np.array([scipy.imread(path) for path in glob(Orig_Path)])/255
    X_orig = []
    for i in range(len(X_Orig)):
        for _ in range(random_restarts):
            X_orig.append(X_Orig[i,:,:,:])
    return np.array(X_orig), np.array(X_Orig)   

def loading_image_generator(dataset):
    
	#if dataset == "celeba":    
    gen = CelebAGenerator()
    gen.GenerateModel()
    gen.LoadWeights()
    G = gen.GetModels()
    channels=3
    L  = 64
    return G
	
	
def masked_model(image,mask, indexes, noise):
            mask_exp_e=mask
            xG_tf_i = tf.cast(image,tf.complex64)
            xG_tf_masked = xG_tf_i*mask_exp_e
            xG_tf_i1 = tf.fft2d(tf.transpose(xG_tf_masked,perm=[0,3,1,2]))
            xG_tf_i11 = tf.transpose(xG_tf_i1,perm=[0,2,3,1])
            xG_tf_phase = tf.reshape(xG_tf_i11,[64*64*3,-1])
            xG_tf_blurry_phase  = tf.gather(xG_tf_phase,indexes)
            var =  tf.abs(xG_tf_blurry_phase)
            return var 
        
#def prgan+_masked(): 
	
def mask_type(mask_type, X_Orig):
            if mask_type == 'binary':
                mask = np.random.choice([0,1], size = (X_Orig.shape[0],64,64,3)).astype('complex64')
            else:
                mask = np.exp(1j*2*np.pi*np.random.randn(X_Orig.shape[0],64,64,3)).astype('complex64')
            return mask



