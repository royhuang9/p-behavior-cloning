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

The first layer is normalization, which normalizes every pixel to [-1  1]. The second, third, fourth, fifth and sixth layers are convnets which have the filter size (24,5,5),(36,5,5),(48,5,5),(64,3,3),(64,3,3). There is a batch normalization layer is append in every convnets. Except the last convnet, all convnets have stride of 2*2. The first convet also include a maxpooling sublayer. Relu activation is used.

There are four fully connected neuron network layers followed. Every fully connected neuron network layers has a dropout and batch normaliation sublayers.

Batch normalization is used to speed convergence. Dropout and L2 regularization for weights are used for overfitting.

A python generator is implemented to provide images data for training that will save memory.


## Data set and Training:
I included the Udacity data in my dataset. I also drived the car in simulator to collect data with a joystick. The joystick is a cheap one and doesn't operate smoothly, but it is better than keyboard. Another explanation about the joystick is that I am not good at playing game, so I cannot operate the car smoothly even with a joystick.

Most time I drived the car in the middle of track to collect data. According to Udacity notes, I need to drive the car to the road side to collect some recovery data. But I made a mistake here. I collect both the car weave off the road and back the middle of road. When I realize this mistake, I already mix all data together and I cannot distinguished them. Eventually I collect 82011 images which including left, center and right images.

The image size is 160x320, I didn't resize or crop the image. The first layer of model will normalize image. In my training, left and right images are also used. In the image generator, the steering angle of left image is shift [0 0.2] angle randomly. The steering angle of right image is shift [-0.2 0] angle randomly. All images are flipped left to right which will double the images to about 160000. 90% of them are used as traing data and 10% are vlidation data.

< insert some images with steering angle>

Adam optimizer is used. The learning rate is default, 0.001. But when load an old model to refine, the learing rate will be 0.0001, otherwise the new data will mess the model.

Validation data is used to check whether the traing should stop. Keras.callback.EarlyStop is defined with parameters, min_delta=0.002, patience=5. This mean when validation loss doesn't descrease 0.002 at last 5 epoches, the training will stop. Model weights is saved after every epoch. After about 50 epoches, it stopped. I haven't written down the exactly number. At least it is bigger than 5 epoches that some guys mentioned in slack and medium post.

I ever spent some time to discover how many epoch should be choosed to get a best result. <insert a loss history picture here>. I found both train loss and validation loss doesn't change after some epoches.
![loss history1](/images/loss_history1.png?raw=true "loss history2")

I tried the model I got in autonomous mode. It can pass the bridge and the left turn behind bridge, but run off road and drowned in the flowing right trun.

My solution is refining the model I've got. I collected new data just around the turn where the car run off road. I visualized every image and corrected the steering angle between [0.2, 0.8] manually. Then load the old model, adjust learning rate to 0.0001 and train the model with just new data for only 5 epoch. With this new model, the car can run serveral circles and is stuck on bridge eventually.

I spent a whole day to fix the bridge. Most time it became worse and even cannot pass the first turn. This is an error and try. At last I realized the correct recovery training, only record data when the car is driving from the side of the road back toward the center line. With a new refinement of bridge recovery data, the car can succeed to run on track for many hours until I close the simulator. 


Next Step:
I will try the right track in the simulator. In order to do so, I will improve my mode as following:
Crop the image to abandon the top 1/3 part of the images which have no relationship with drving. 
Convert the iamge to grayscale.

