# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt    
from keras.preprocessing.image import ImageDataGenerator   
from tensorflow.keras.callbacks import EarlyStopping     


IMAGE_SIZE = [224, 224]
vgg = VGG19(input_shape=(224, 224, 3),
                                  weights='imagenet',
                                  include_top=False)

for layer in vgg.layers:
  layer.trainable = False
  
x = Flatten()(vgg.output)
prediction = Dense(3, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)




train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 45,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   vertical_flip = True,
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/gdrive/MyDrive/UntitledFolder/rps',
                                                 target_size = (224, 224),
                                                 batch_size = 25,
                                                 shuffle = True,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/gdrive/MyDrive/UntitledFolder/rps-test-set',
                                            target_size = (224, 224),
                                            batch_size = 25,
                                            class_mode = 'categorical')


r = model.fit(
  training_set,
  validation_data=test_set,     
  epochs=5,
  steps_per_epoch=len(training_set),
  verbose=1,
  validation_steps=len(test_set),

)

# loss    callbacks=custom_callbacks
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


from keras.models import load_model

model.save('model.h5')
