# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:26:15 2024

@author: Poll
"""

#Import packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
#%% Define folders and load data
base_dir= r"C:/Users/Poll/Documents/Fall2024/NN_final/dataverse_files/Classification"
train_dir= f"{base_dir}/training"
test_dir = f"{base_dir}/test"
val_dir= f"{base_dir}/validation"

resize = (224, 224)
batch_size = 32

train_dataset= tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=resize,
    batch_size=batch_size,
    label_mode="categorical"  
)


test_dataset= tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=resize,
    batch_size=batch_size,
    label_mode="categorical"
)

val_dataset=tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=resize,
    batch_size=batch_size,
    label_mode="categorical"
)

class_names = train_dataset.class_names 

#print("Class Names:", class_names)

# Normalization
normalization_layer= tf.keras.layers.Rescaling(1./255)
train_dataset= train_dataset.map(lambda h, w: (normalization_layer(h), w))
test_dataset= test_dataset.map(lambda h, w: (normalization_layer(h), w))
val_dataset= val_dataset.map(lambda h, w: (normalization_layer(h), w))
#for images, labels in train_dataset.take(1):
#    print(images.shape)  
#    print(labels.shape) 
#%% CNN Model
CNN_Model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dense(128, activation='relu'),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')  ])

CNN_Model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

CNN_Model.summary()
                 
Hist_CNN=CNN_Model.fit(train_dataset, epochs=7, validation_data=val_dataset)
#%%
CNN_loss, CNN_accuracy = CNN_Model.evaluate(test_dataset, verbose=2)
print("The test accuracy is \n", CNN_accuracy)
CNNpredictions=CNN_Model.predict(test_dataset)
print(f"CNN Model Test Loss: {CNN_loss:.4f}")
print(f"CNN Model Test Accuracy: {CNN_accuracy:.4f}")


print("The predictions, test_dataset are \n", CNNpredictions)
print("The shape of the predictions, test_dataset is \n", CNNpredictions.shape) 
print("The single prediction vector for test_dataset[2] is \n", CNNpredictions[2]) 
print("The max - final prediction label for test_dataset[2] is\n", np.argmax(CNNpredictions[2])) 
#%%
plt.plot(Hist_CNN.history['accuracy'], label='accuracy')
plt.plot(Hist_CNN.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Accuracy vs Epochs',fontsize=20)
plt.legend(loc='lower right',fontsize=20)
#%%
plt.plot(Hist_CNN.history['loss'], label='loss')
plt.plot(Hist_CNN.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Loss vs Epochs',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
#%% Create labels test images
test_images_CNN, test_labels_CNN = [], []

for testimages_CNN, testlabels_CNN in test_dataset:
    test_images_CNN.extend(testimages_CNN.numpy())
    test_labels_CNN.extend(testlabels_CNN.numpy())

test_images_CNN = np.array(test_images_CNN)
test_labels_CNN = np.array(test_labels_CNN)
#%% Create true labels and confusion matrix
predicted_labels_CNN = np.squeeze(np.array(CNNpredictions.argmax(axis=1)))
true_labels_CNN = np.argmax(test_labels_CNN, axis=1) 
CNN_CM=confusion_matrix(predicted_labels_CNN, true_labels_CNN)
print("The confusion matrix is \n", CNN_CM)   

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(CNN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: CNN') 
#%% Create labels train and test images for ANN.
train_images_ANN, train_labels_ANN = [], []

for trainimages_ANN, trainlabels_ANN in train_dataset:
    train_images_ANN.extend(trainimages_ANN.numpy())
    train_labels_ANN.extend(trainlabels_ANN.numpy())

train_images_ANN = np.array(train_images_ANN)
train_labels_ANN = np.array(train_labels_ANN)

print("train_images shape:", train_images_ANN.shape) 
print("train_labels shape:", train_labels_ANN.shape)  

test_images_ANN, test_labels_ANN = [], []

for testimages_ANN, testlabels_ANN in test_dataset:
    test_images_ANN.extend(testimages_ANN.numpy())
    test_labels_ANN.extend(testlabels_ANN.numpy())

test_images_ANN = np.array(test_images_ANN)
test_labels_ANN = np.array(test_labels_ANN)
#%% ANN Model
train_images_flattened = train_images_ANN.reshape(-1, 224 * 224 * 3)
test_images_flattened = test_images_ANN.reshape(-1, 224 * 224 * 3)
print("Updated train_labels shape:", train_labels_ANN.shape)
print("Updated test_labels shape:", test_labels_ANN.shape)

My_ANN_Model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150528,)),  
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(64, activation='relu'),   
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(64, activation='relu'),   
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(64, activation='relu'),  
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(64, activation='relu'),    
    tf.keras.layers.Dense(4, activation='softmax')  ])

My_ANN_Model.summary()
My_ANN_Model.compile(
    loss='categorical_crossentropy',  
    optimizer='adam',                
    metrics=['accuracy']             
)
print("train_images_flattened shape:", train_images_flattened.shape) 
print("test_images_flattened shape:", test_images_flattened.shape)  
print("train_labels shape:", train_labels_ANN.shape)  
print("test_labels shape:", test_labels_ANN.shape)  

Hist_ANN=My_ANN_Model.fit(train_images_flattened, train_labels_ANN, epochs=7, validation_data=(test_images_flattened,test_labels_ANN))
#%%
ANN_loss, ANN_accuracy = My_ANN_Model.evaluate(test_images_flattened, test_labels_ANN, verbose=2)
ANNpredictions=My_ANN_Model.predict([test_images_flattened])
#%%
print(f"ANN Model Test Loss= {ANN_loss:.4f}")
print(f"ANN Model Test Accuracy= {ANN_accuracy:.4f}")

print("The predictions, test_images_flattened are \n", ANNpredictions)
print("The shape of the predictions, test_images_flattened is \n", ANNpredictions.shape) 
print("The single prediction vector for test_images_flattened[2] is \n", ANNpredictions[2]) 
print("The max - final prediction label for test_images_flattened[2] is\n", np.argmax(ANNpredictions[2])) 
#%%
plt.plot(Hist_ANN.history['accuracy'], label='accuracy')
plt.plot(Hist_ANN.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Accuracy vs Epochs',fontsize=20)
plt.legend(loc='lower right',fontsize=20)
#%%
plt.plot(Hist_ANN.history['loss'], label='loss')
plt.plot(Hist_ANN.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Loss vs Epochs',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
#%%
test_images_ANN, test_labels_ANN = [], []

for testimages_ANN, testlabels_ANN in test_dataset:
    test_images_ANN.extend(testimages_ANN.numpy())
    test_labels_ANN.extend(testlabels_ANN.numpy())

test_images_ANN = np.array(test_images_ANN)
test_labels_ANN = np.array(test_labels_ANN)

print("train_images shape:", train_images_ANN.shape) 
print("train_labels shape:", train_labels_ANN.shape)  
predicted_labels = np.squeeze(np.array(ANNpredictions.argmax(axis=1)))
true_labels_ANN = np.argmax(test_labels_ANN, axis=1) 
ANN_CM=confusion_matrix(predicted_labels, true_labels_ANN)
print("The confusion matrix is \n", ANN_CM)   

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(ANN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: ANN') 
#%% Recurrent CNN
RCNN_Model = models.Sequential([  
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Reshape((-1, 128)),  
    layers.LSTM(128, return_sequences=True),  
    layers.Dense(128, activation='relu'),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')  ])

RCNN_Model.compile(optimizer='adam',
                   loss=tf.keras.losses.CategoricalCrossentropy(),
                   metrics=['accuracy'])
RCNN_Model.summary()
#%%
Hist_RCNN = RCNN_Model.fit(train_dataset, epochs=7, validation_data=val_dataset)
RCNN_loss, RCNN_accuracy = RCNN_Model.evaluate(test_dataset, verbose=2)
print(f"RCNN Model Test Loss: {RCNN_loss:.4f}")
print(f"RCNN Model Test Accuracy: {RCNN_accuracy:.4f}")
RCNN_predictions = RCNN_Model.predict(test_dataset)
predicted_labels_RCNN = np.squeeze(np.array(RCNN_predictions.argmax(axis=1)))
#%% Create test image labels RCNN
test_images_RCNN, test_labels_RCNN = [], []

for test_images, test_labels in test_dataset:
    test_images_RCNN.extend(test_images.numpy())
    test_labels_RCNN.extend(test_labels.numpy())

test_labels_RCNN = np.array(test_labels_RCNN)
true_labels_RCNN = np.argmax(test_labels_RCNN, axis=1)
#%%
plt.plot(Hist_RCNN.history['accuracy'], label='Train Accuracy')
plt.plot(Hist_RCNN.history['val_accuracy'], label='Validation Accuracy')
plt.title('RCNN Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(Hist_RCNN.history['loss'], label='Train Loss')
plt.plot(Hist_RCNN.history['val_loss'], label='Validation Loss')
plt.title('RCNN Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%% Confusion matrix
RCNN_CM = confusion_matrix(predicted_labels_RCNN, true_labels_RCNN)
print("Confusion Matrix: \n", RCNN_CM)
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(RCNN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True Labels')
ax.set_ylabel('Predicted Labels')
ax.set_title('Confusion Matrix: RCNN')
plt.show()