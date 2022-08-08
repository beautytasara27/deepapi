
# Transfer learning for Fruit Image Classification. (Sorting based on appearance).
import cv2
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#directories
train_dir = "/home/madonnacarter1/SplitData/train"
val_dir = "/home/madonnacarter1/SplitData/val"
test_dir = "/home/madonnacarter1/SplitData/test"
save_dir = "/home/madonnacarter1/saved_models/"

#constants
batchSize = 128
stepsPerEpoch = 100
validationSteps = 35
validationBatchSize=32
target_size = (224, 224)
epoch = 120
inputs = (224, 224, 3)
Inputs = keras.Input(shape=inputs)

#change color Space
def Change_Color(image):
    image = np.array(image)
    new_image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    return new_image

#apply gaussian blur
def Gaussian_Filter(image):
    blurred_image = cv2.GaussianBlur(image,(5,5),0)
    return blurred_image

#generate data
train_datagen = ImageDataGenerator(rescale = 1./255., zoom_range=0.1, height_shift_range=0.1, vertical_flip=True, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale = 1./255., zoom_range=0.1, height_shift_range=0.1, vertical_flip=True, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255., zoom_range=0.1, height_shift_range=0.1, vertical_flip=True, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=batchSize, shuffle=True, class_mode = 'categorical', target_size = target_size)

validation_generator = val_datagen.flow_from_directory(val_dir, batch_size=validationBatchSize, shuffle=True, class_mode = 'categorical', target_size = target_size)

test_generator = test_datagen.flow_from_directory(test_dir, batch_size=batchSize, shuffle=True, class_mode = 'categorical', target_size = target_size)

#show samples
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(15):
        ax = plt.subplot(5,5, n+1)
        arr = np.asarray(np.squeeze(image_batch[n]))
        plt.imshow(arr, vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
sample_train_images, lbl = next(train_generator)
show_batch(sample_train_images, lbl)

def Load_MobileNetV3():
    mobilenet_model = MobileNetV3Small(include_top=False,weights='imagenet',input_shape=inputs,pooling='max')
    for layer in mobilenet_model.layers[:-10]:
        layer.trainable = False
    output = mobilenet_model(Inputs)
    output = keras.layers.Flatten()(output)
    fullyConnected1 = Dense(15, activation='softmax')(output)
    return keras.Model(inputs=Inputs, outputs=fullyConnected1)
MobilenetV3_Model = Load_MobileNetV3()
MobilenetV3_Model.summary()

def Load_ResNet50():
    res_model = ResNet50(include_top=False,weights="imagenet",pooling='max',input_shape=inputs)
    for layer in res_model.layers:
        layer.trainable = False
    output = res_model(Inputs)
    output = keras.layers.Flatten()(output)
    fullyConnected1 = Dense(15, activation='softmax')(output)
    return keras.Model(inputs=Inputs, outputs=fullyConnected1)
Resnet50_Model = Load_ResNet50()
Resnet50_Model.summary()

def Load_VGG16():
    vgg_model = VGG16(include_top=False,weights='imagenet',input_shape=inputs,pooling='max')
    for layer in vgg_model.layers:
        layer.trainable = False
    output = vgg_model(Inputs)
    output = keras.layers.Flatten()(output)
    fullyConnected1 = Dense(15, activation='softmax')(output)
    return keras.Model(inputs=Inputs, outputs=fullyConnected1)
Vgg16_Model = Load_VGG16()
Vgg16_Model.summary()

def Compile_Fit(model, path):
    model = model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['acc'])
    model = model.fit(train_generator, steps_per_epoch = stepsPerEpoch, epochs = epoch, validation_data=validation_generator, validation_steps=validationSteps)
    # Save the entire model as a SavedModel.
    model.save(path+"vgg1")
    with open(path, 'wb') as file_pi:
            pickle.dump(model.history.history+"history_vgg1", file_pi)
    return model

model = Compile_Fit(Vgg16_Model, save_dir) #pass vgg/mobilenet/resnet

reloaded_model = keras.models.load_model(save_dir +'vgg_model100')
with open('/home/madonnacarter1/history/history_vgg100', 'rb') as f:
    reloaded_history = pickle.load(f)

#pass the model if still in memory else pass reloaded_model
def Predict(model):
    Y_pred = model.predict(test_generator,batch_size=32 )
    y_pred = np.argmax(Y_pred, axis=1)
    return y_pred
y_pred = Predict(reloaded_model)

def Plot_Confusion_Matrix():
    cm = confusion_matrix(test_generator.classes, y_pred)
    class_labels = list(test_generator.class_indices.keys())
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    print(classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys()))
Plot_Confusion_Matrix()

def Plot_Accuracy_Loss(history):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=False)
    ax1.set_title('Accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.plot(history['acc'], label='training')
    ax1.plot(history['val_acc'], label='validation')
    ax2.set_title('Loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.plot(history['loss'], label='training')
    ax2.plot(history['val_loss'], label='validation')
    plt.show()
Plot_Accuracy_Loss(reloaded_history)

