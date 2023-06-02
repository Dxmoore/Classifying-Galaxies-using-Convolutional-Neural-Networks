import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

#import data
input_data, labels = load_galaxy_data()

#check out input and labels dimensions
print(input_data.shape)
print(labels.shape)

#Because the last dimension of the data is 3, you know that the image data is RGB/in color. Because the last dimension of the labels is 4, and there are four classes, you know that the labels are one-hot vectors. For example, [1,0,0,0] → Normal galaxy.

#split the data
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size = .2, random_state = 222, stratify = labels)


#Preprocess data
data_generator = ImageDataGenerator(rescale = 1/255)

#create arrays to iterator through training and test data
training_iterator = data_generator.flow(x_train, y_train,batch_size=5)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

#build the model

#instantiate
model = tf.keras.Sequential()

#input layer, shape based on size of images and number of chanels for color (RGB)
#model includes  two convolutional layers, interspersed with max pooling layers, followed by two dense layers
model.add(tf.keras.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Dense(4,activation="softmax")) #output
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))
#compile the model with Adam optimizer and crossentropy for the labels due to one-hot encoding, scoring metrics using AUC and Accuracy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])


#view the dimensions of the model
model.summary()

#fit and train the model
model.fit(
        training_iterator,
        steps_per_epoch=len(x_train)/5,
        epochs=8,
        validation_data=validation_iterator,
        validation_steps=len(x_valid)/5)

#What do these results mean? Your accuracy tells you that your model assigns the highest probability to the correct class more than 60% of the time. For a classification task with over four classes, this is no small feat: a random baseline model would achieve only ~25% accuracy on the dataset. Your AUC tells you that for a random galaxy, there is more than an 80% chance your model would assign a higher probability to a true class than to a false one.

#Visualize the data
from visualize import visualize_activations

visualize_activations(model,validation_iterator)


#This function loads in a sample batch of data using your validation iterator.It uses model.predict() to generate predictions for the first sample images. Next, it compares those predictions with the true labels and prints the result. It then saves the image and the feature maps for each convolutional layer using matplotlib.
