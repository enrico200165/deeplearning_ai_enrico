import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline

batch_size = 16
nr_epochs = 10

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# ---------------------------------------------------------
#         Happy Model
# ----------------------------------------------------------
# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    print("input_shape param = ", input_shape)
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32 # filters: Integer, the dimensionality of the output space
               , (7, 7) # kernel_size: An integer or tuple/list of 2 integers,
               #specifying the height and width of the 2D convolution window.
               # Can be a single integer to specify the same value for all spatial dimensions.
               , strides = (1, 1) # strides: An integer or tuple/list of 2 integers,
               # specifying the strides of the convolution along the height and width.
               # Can be a single integer to specify the same value for all spatial dimensions.
               , name = 'conv0' # name: An optional name string for the layer.
               # Should be unique in a model (do not reuse the same name twice).
               # It will be autogenerated if it isn't provided.
               )(X)

    X = BatchNormalization(
        axis = 3 # axis: Integer, the axis that should be normalized (typically the features axis).
        # For instance, after a Conv2D layer with data_format="channels_first",
        # set axis=1 in BatchNormalization.
        # channels_first corresponds to inputs with shape (batch, channels, height, width).
        # channels_last corresponds to inputs with shape (batch, height, width, channels)
        , name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2) # pool_size: integer or tuple of 2 integers,
                     # factors by which to downscale (vertical, horizontal).
                     # (2, 2) will halve the input in both spatial dimension.
                     # If only one integer is specified, the same window length will be used for both dimensions.
                     , name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1 # units: Positive integer, dimensionality of the output space.
              , activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')


    ### END CODE HERE ###

    return model

### START CODE HERE ### (1 line)
# A shape tuple (integer), not including the batch size
# EV (600, 64, 64, 3) -> (64, 64, 3)
happyModel = HappyModel( (64, 64, 3) )
### END CODE HERE ###


### START CODE HERE ### (1 line)
ret = happyModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("outcome of fit()", ret)
### END CODE HERE ###

### START CODE HERE ### (1 line)
h = happyModel.fit(X_train, Y_train, batch_size=batch_size
                   ,epochs= nr_epochs, verbose=1)
print("History of happyModel.compile()", h)
### END CODE HERE ###

### START CODE HERE ### (1 line)
preds = happyModel.predict(X_test)
### END CODE HERE ###
print("first predictions", preds[:10])


model_evaluation = happyModel.evaluate(X_test, Y_test, batch_size=16)
metrics = happyModel.metrics_names
for i in range(len(model_evaluation)):
    print("{} = {}".format(metrics[i],model_evaluation[i]))

print()