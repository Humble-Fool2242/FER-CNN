# FER-CNN
Facial Emotion Recognition using Convolutional Neural Network

# Abstract:
Facial expression recognition is a topic of great
interest in most fields from artificial intelligence and gaming to
marketing and healthcare. The goal of this paper is to classify
images of human faces into one of seven basic emotions. A
number of different models were experimented with, including
decision trees and neural networks before arriving at a final
Convolutional Neural Network (CNN) model. CNNs work better
for image recognition tasks since they are able to capture spacial
features of the inputs due to their large number of filters. The
proposed model consists of six convolutional layers, two max
pooling layers and two fully connected layers. Upon tuning of the
various hyperparameters.


# Introduction:
Facial emotion recognition could be
used in conjunction with other systems to provide a form of
safety. For instance, ATMs could be set up such that they won’t
dispense money when the user is scared. In the gaming industry, 
emotion-aware games can be developed which could vary
the difficulty of a level depending on the player’s emotions. It
also has uses in video game testing. At present, players usually
give some form of verbal or written feedback. Using facial
emotion recognition, the number of testers can be increased to
accomodate people who use different languages or people who
are unable to cohesively state their opinions on the game. By
judging their expressions during different points of the game,
a general understanding of the game’s strong and weak points
can be discerned. Emotions can also be gauged while a viewer
watches ads to see how they react to them. This is especially
helpful since ads do not usually have feedback mechanisms
apart from tracking whether the ad was watched and whether
there was any user interaction. Software for cameras can use
emotion recognition to take photos whenever a user smiles.

# Dataset:
The Facial Expression dataset (FER-2013) was chosen for training of our model. The FER-2013 dataset was introduced in the ICML 2013
Challenges in Representation Learning [2]. It contains 35,887
images with the following basic expressions: angry, disgusted,
fearful, happy, sad, surprised and neutral.


# Convolutional Neural Network:
A Convolutional neural network is a neural network comprised of convolution layers which does computational heavy
lifting by performing convolution. Convolution is a mathematical operation on two functions to produce a third function. It
is to be noted that the image is not represented as pixels,
but as numbers representing the pixel value. In terms of
what the computer sees, there will simply just be a matrix
of numbers. The convolution operation takes place on these
numbers. We utilize both fully-connected layers as well as
convolutional layers. In a fully-connected layer, every node is
connected to every other neuron. They are the layers used
in standard feedforward neural networks. Unlike the fullyconnected layers, convolutional layers are not connected to
every neuron. Connections are made across localized regions.
A sliding ”window” is moved across the image. The size
of this window is known as the kernel or the filter. They
help recognise patterns in the data. For each filter, there are
two main properties to consider - padding and stride. Stride
represents the step of the convolution operation, that is, the
number of pixels the window moves across. Padding is the
addition of null pixels to increase the size of an image. Null
pixels here refers to pixels with value of 0. If we have a
5x5 image and a window with a 3x3 filter, a stride of 1
and no padding, the output of the convolutional layer will
be a 3x3 image. This condensation of a feature map is known
as pooling. In this case, ”max pooling” is utilized. Here, the
maximum value is taken from each sliding window and is
placed in the output matrix.
Convolution is very effective in image recognition and
classification compared to a feed-forward neural network.
This is because convolution allows to reduce the number
of parameters in a network and take advantage of spatial
locality. Further, convolutional neural networks introduce the
concept of pooling to reduce the number of parameters by
downsampling. Applications of Convolutional neural networks
include image recognition, self-driving cars and robotics. CNN
is popularly used with videos, 2D images, spectrograms,
Synthetic Aperture Radars.



# Final Model architecture:
The final model is depicted in Table IV.
The network consists of six two-dimensional convolutional
layers, two max pooling layers and two fully connected layers.
Max pooling uses the maximum value from each of a cluster
of neurons at the prior layer. This reduces the dimensionality
of the output array. The input to the network is a preprocessed
face of 48 x 48 pixels. It
was decided to go with a deeper network over a wide one.
The advantage of using more layers is that it prevents memorization. 
A wide but shallow network memorizes well but
does not generalize well. Multi-layer networks learn features
at levels of abstractions allowing them to generalize well. The
number of layers were selected so as to maintain a high level of
accuracy while still being fast enough for real-time purposes.
The proposed CNN differs from a simple CNN in that it uses 4
more convolutional layers and each of its convolutional layers
differ in filter size. In addition, it utilized max pooling and
dropout more effectively in order to minimize overfitting.
         
         TABLE IV
  ARCHITECTURE OF THE PROPOSED CNN
     Proposed Convolutional Neural Network
         CONV2D-64
          RELU
          CONV2D-64
          RELU
          MAXPOOL2D
          DROPOUT
          CONV2D-128
          RELU
          CONV2D-128
          RELU
          CONV2D-256
          RELU
          CONV2D-256
          RELU
          MAXPOOL2D
          DROPOUT
          FLATTEN
          FULLY CONNECTED
          RELU
          DROPOUT
          FULLY CONNECTED
          SOFTMAX
          
          
The network consists of two convolutional layers with a
filter size of 64 each. This is then followed by a max pooling
layer. A dropout of rate 0.25 is applied to reduce overfitting.
This is followed by a sequence of four convolutional layers.
The first two have a filter size of 128 each and the latter two
have a filter size of 256 each. A single max pooling layer
follows these four layers with a dropout of rate 0.25. In order
to convert the output into a single dimensional vector, the
output of the previous layers was flattened. A fully connected
layer with a L2 regularizer of penalty of 0.001 is then used
alogn with an additional dropout of rate 0.5. Finally, a fully
connected layer with a softmax activation function serves as
the output layer.
The kernel size, that is, the width and height of the 2D
convolutional window is set to 3 x 3 for all convolutional
layers. Each max pooling layer is two dimensional and uses a
pool size of 2 x 2. This halves the size of the output after each
pooling layer. All the layers bar the output layer used a ReLU
activation function. The ReLU activation function is used here
due to benefits such as sparsity and a reduced likelihood of
vanishing gradient. The softmax activation function was used
in the final output layer to receive the predicted probability of
each emotion.
This model provided a base accuracy of 0.55 on the testing
set. The hyperparamters were then tuned, namely the batch
size, the optimizer and the number of epochs. Each model was
set to run for 100 epochs. However, in the interest of saving
time and computational power, the network was allowed to
stop training if there was no change in the accuracy over
consecutive epochs. That is, the network would stop training if
there was no change in the accuracy over 4 continuous epochs.
This saved both time and computational power, especially in
cases where there was no change in the accuracy within the
earlier epochs themselves. The decision turned out to be a
good one as none of the models exceeded 20 epochs.

# Testing:
The dataset was initially split into an 80%-training set and a
20%-testing set. During the testing phase, each of the trained
networks was loaded and fed the entire testing set one image at
a time. This image was a new one which the model had never
seen before. The image fed to the model was preprocessed in
the same way as detailed in ??. Thus the model did not know
already what the correct output was and had to accurately
predict it based on its own training. It attempted to classify
the emotion shown on the image simply based on what it had
already learned along with the characteristics of the image
itself. Thus in the end, it gave a list of classified emotion
probabilities for each image. The highest probability emotion
for each image was then compared with the actual emotions
associated with the images to count the number of accurate
predictions.
The accuracy formula is detailed below. It simply counts
the number of samples where the model correctly predicted
the emotion and divides it by the total number of samples in
the testing set. Here, the testing set consists of about 7,178
images.
Accuracy =Num.CorrectlyP redictedEmotions/TotalNum.Samples 

# RESULTS
Upon tuning the hyperparameters, the highest accuracy was
achieved for each optimizer. Using the RMSProp optimizer,
an accuracy of 0.57 was reached over 20 epochs and a batch
size of 96. The Stochastic Gradient Descent optimizer gave an
accuracy of 0.55 out of the box and it could not be increased
significantly by further tuning of the hyperparameters. Using
the Adam optimizer with the default settings, a batch size
of 64 and 10 epochs lead to an astoundingly low accuracy of
0.17. However upon setting the learning rate to 0.0001 and the
decay to 10e − 6, the highest accuracy of 0.60 was attained.
A comparison of the various hyperparameters that were tuned
can be seen in Table V.
TABLE V
COMPARISON OF HYPERPARAMETERS:

Optimizer  Batch Size Epochs Accuracy
RMSProp     64         24    55.96%
RMSProp     32         9     42.07%
RMSProp     96         20    57.39%
SGD         64         10    55.90%
Adam        64         10    17.38%
Adam        128        20    60.58%

Based on these results it can be concluded that the Adam
optimizer which initially provided an abysmal accuracy turned
out to be the best fit for the data


