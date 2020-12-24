#%%
# Convolutional Neural Network

#%%
# Libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#%%
# Data Preprocessing

#%%
# Preprocessing the Training set
# Transforming Images to Avoid Overfitting
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#%%
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


#%%
# Building the CNN

#%%
# Initializing the CNN
cnn = tf.keras.models.Sequential()

#%%
# Input Convolution Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#%%
# Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#%%
# Adding Second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#%%
# Flattening
cnn.add(tf.keras.layers.Flatten())

#%%
# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#%%
# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


#%%
# Training the CNN

#%%
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#%%
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


#%%
# Making A Prediction

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

# Check indices from the result of CNN Training
training_set.class_indices

# Making Prediction
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)