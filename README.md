# p-behavior-cloning
Udacity self-drving car project 2

This project is a regression problem with convnet implementation. A trained model need to generate steering angle to operate simulator. The model is trained by images collected from simulator.

##Architect of model:

The nvidia paper was the start point for me, but I added batch normalization layers and L2 regualrization to weights. Of course, the input size of the image is different. I adjusted it accordingly.

Keras summary show below:

Layer (type)                     Output Shape          Param \#     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 80, 160, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 40, 80, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 40, 80, 24)    96          maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 40, 36)    21636       batchnormalization_1[0][0]       
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 20, 40, 36)    144         convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 20, 48)    43248       batchnormalization_2[0][0]       
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 10, 20, 48)    192         convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 10, 48)     20784       batchnormalization_3[0][0]       
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 5, 10, 48)     192         convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 8, 48)      20784       batchnormalization_4[0][0]       
____________________________________________________________________________________________________
batchnormalization_5 (BatchNorma (None, 3, 8, 48)      192         convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           batchnormalization_5[0][0]       
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           activation_1[0][0]               
____________________________________________________________________________________________________
batchnormalization_6 (BatchNorma (None, 100)           400         dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        batchnormalization_6[0][0]       
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           activation_2[0][0]               
____________________________________________________________________________________________________
batchnormalization_7 (BatchNorma (None, 50)            200         dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         batchnormalization_7[0][0]       
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           activation_3[0][0]               
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]                  
====================================================================================================
Total params: 230,563
Trainable params: 229,855
Non-trainable params: 708
____________________________________________

The first layer is normalization, which normalizes every pixel to [0 - 1]. The second, third, fourth, fifth and sixth layers are convnets which have the filter size (24,5,5),(36,5,5),(48,5,5),(64,3,3),(64,3,3). There is a batch normalization layer is append in every convnets. Except the last convnet, all convnets have stride of 2*2. The first convet also include a maxpooling sublayer. Relu activation is used.

There are four fully connected neuron network layers followed. Every fully connected neuron network layers has a dropout and batch normaliation sublayers.

Batch normalization is used to speed convergence. Dropout and L2 regularization for weights are used for overfitting.

A python generator is implemented to provide images data for training that will save memory.


##Training:
Adam optimizer is used. The learning rate is default, but during
