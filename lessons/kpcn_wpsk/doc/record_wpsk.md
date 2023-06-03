# enhance input noisy image(1 spp)

(temporally accumulating consecutive 1-spp)
We employ a temporal accumulation pre- processing step before sending the noisy inputs to the denoising pipeline:
    We first reproject the previous frame to the current frame with the motion vector and then judge their geometry consistency by world position and shad- ing normal feature buffers. Current frame pixels that passed the consistency test are blended with their corresponding pixels in the previous frame, while the failed pixels remain original 1 spp.

    like BMFR?






# ImportantNet use RepVGG block:
two type net:3 layer or 6 layer





# kernel construction process
1.unfold
2.normalize with softmax function




# kernel fusion
with different size kernel size to multiple important maps

filtering kernels of different sizes respond to filtering on different frequencies
    It gives higher weight to large-sized kernel fil- tered results in low-frequency regions




where the per-pixel average weights αi are also predicted by our ImportanceNet in the last convolutional layer and normalized by a softmax activation function:
    ImportNet output: ( alpha map + important map ) * N(=6)


# dataset


Our inputs to the ImportanceNet are comprised of low dynamic range (LDR) 3-channel color modulated with 3-channel albedo, 3- channel normal, and 1-channel depth (10-channel in total).


We linearly scale normal buffer and depth buffer to range [0, 1]. 


gamma correction for the BMFR dataset and Filmic tone mapping for the Tungsten dataset



During training, we **uniformly sample** 80 image patches with resolution 128 × 128 from each 1280 × 720 frame of training data to form a training dataset and sample another 20 patches each frame for the valida- tion dataset.




# train

loss: SMAPE


The weights of our neural network are initialized with the default uniform distribution.


We use a batch size of 64 and train the network with Adam [KB14] optimizer with a learning rate of 0.001.

 For the BMFR dataset, we construct six filtering kernels for kernel fusion module, which means that our ImportanceNet predicts 6 importance maps and six corresponding averaging weight maps. We use base kernel size kb = 3 and kernel size incremental step ks = 2, so our minimum kernel size is 3 and maximum kernel size is 13:
    {3,5,...,13}



For each test case, we hold out one scene as test set and train the network with remaining scenes for 500 epoch








# record

## dataset

1.prepare 60 consecutive frames in one scene
2.preprocess dataset:
write acc color
3.train with acc color dataset!


# code

## data_preprocess.py

output:
acc_colors


TODO:
change 60 to 3