import numpy as np
import tensorflow as tf
import scipy.io as sci
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import numpy as np
#from Datasets import *
from CelebAGenerator import *
from ShoeGenerator import *
K.set_learning_phase(0)
from glob import glob
import os
import numpy as np
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim
import scipy.misc as scipy
from MotionBlurGenerator import *
from utils import *


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--measurements", type = int, default=12288, help="Path to the document image")
    #parser.add_argument("-n", "--noise", default="..", help="Path to store the result")
    parser.add_argument("-d", "--dataset", default='celeba', help="datasets")
    parser.add_argument("-r", "--random", type = int, default=15, help="datasets")
    parser.add_argument("-mod","--model",default = 'mask_fourier',help = 'model (mask_fourier or fourier)')
    parser.add_argument("-p","--pad",type = int, default = 64, help = 'padding for fourier model')
    return parser.parse_args()

def main():
    args = args_processor()
    dataset = args.dataset
    measurements = args.measurements
    random_restarts = args.random 
    model = args.model
	
    Orig_Path   = './original_images/%s/*.png'%(dataset)# CHECKING IF SAVE DIR EXISTS
    X_Orig, orig = loading_images(Orig_Path,random_restarts) 
    
    ########################
    rand_indexes = np.random.choice(np.arange(12288),[1,measurements],replace=False).reshape(measurements)
    mask = mask_type('binary',X_Orig)
    #forward model for numpy
    G = loading_image_generator(dataset)
    

    BP_images = masked_model(X_Orig,mask,rand_indexes, noise=0)

    sess = tf.InteractiveSession()
    BP_Images = BP_images.eval()
    sess.close()
    
    #tensorflow part
    z_tf  = tf.Variable(tf.random_normal(shape=(X_Orig.shape[0], 100)))
    Y_tf  = tf.placeholder(dtype="float32", shape=(measurements,X_Orig.shape[0])) #PUT Y HERE AND CHANGE SHAPE
    mask_exp = tf.placeholder(dtype="complex64", shape=(X_Orig.shape[0],X_Orig.shape[1],X_Orig.shape[2],X_Orig.shape[3]))
    xG_tf_i  = G(z_tf)#[0,:,:,:]
    xG_tf_i =  (xG_tf_i +1)/2
    
    mask_exp_e = mask_exp#tf.expand_dims(mask_exp,axis=0)

    xG_tf_blurry_phase = masked_model(xG_tf_i,mask_exp_e,rand_indexes, noise=0)

    
    Loss_tf             = tf.reduce_mean(tf.square(Y_tf - xG_tf_blurry_phase),axis=0)
    
    LEARNING_RATE =  0.005   #0.001
    steps          = 5000
    optimizer       = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    opt                 = optimizer.minimize(Loss_tf, var_list=[z_tf])
    sess = K.get_session()
    sess.run(tf.variables_initializer([z_tf]))
    Loss = []
    for i in tqdm(range(steps)):
        kk , loss = sess.run([opt,Loss_tf], feed_dict={Y_tf:BP_Images, mask_exp:mask}) #put Y and A here too
        if (i %100) == 0:
            x_hat= sess.run([xG_tf_i])
            x_hat = np.array(x_hat)
            #print(compare_psnr(X_Orig[0],x_hat[0,:,:,:].astype('float64')))
            #plt.imshow(x_hat[0,:,:,:].astype('float64'))
            #plt.subplot(1,5,1)
            #plt.show() 
            Loss.append(np.mean(loss))
            
    #extracting best images from random restarts
    x_hat = x_hat[0,:,:,:,:]
    X_Hat = []
    #W_Hat = []

    for i in range(len(orig)):
        mini = loss[i*random_restarts:(i+1)*random_restarts].argmin() + i*random_restarts
        x_i = x_hat[mini]
        #w_i = w_hat[mini]
        X_Hat.append(x_i)
        #W_Hat.append(w_i)

    X_hat = np.array(X_Hat)
    #W_hat = np.array(W_Hat)
    
    PSNR = []
    SSIM = []
    for i in range(len(orig)):
        PSNR.append(compare_psnr(orig[i],X_Hat[i].astype('float64')))
        SSIM.append(compare_ssim(orig[i],X_Hat[i].astype('float64'),multichannel = True))
    
    PSNR = np.mean(np.array(PSNR))
    print(PSNR)
    SSIM = np.mean(np.array(SSIM))
    print(SSIM)

    
    SAVE_PATH = './results/celeba/bdpr' + ' - '+str(measurements) + '_meas_' +str(random_restarts) + 'RR'
    try:
        os.stat(SAVE_PATH)
    except:
        os.mkdir(SAVE_PATH)
        
    #saving to folder
    for i in range(len(orig)):
        scipy.imsave(SAVE_PATH+'/'+'im_'+str(measurements) + '_meas_' +str(random_restarts) + '_RR_'+str(i)+'.png',X_Hat[i])
        #scipy.imsave(SAVE_PATH+'/'+'bl_'+str(m) + '_meas_' +str(random_restarts) + '_RR_'+str(i)+'.png',W_Hat[i])
    
        
if __name__ == "__main__":
	main()

