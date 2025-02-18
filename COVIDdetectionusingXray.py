import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np

# Define paths for train and test datasets yaha direct daal diya 
train_dir = r'D:\new\Covid19-dataset\train'
test_dir = r'D:\new\Covid19-dataset\test'

# Check directories are correct
print("Training set directories:", os.listdir(train_dir))
print("Test set directories:", os.listdir(test_dir))

# cnn Model building bnayaa
Classifier = Sequential()

# Adding first Convolution Layer and Pooling (1st layer)
Classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second Convolution Layer and Pooling (2nd layer)
Classifier.add(Conv2D(32, (3, 3), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))

#\
Classifier.add(Flatten())

# Adding fully connected layer sari layers
Classifier.add(Dense(units=104, activation='relu'))

# Adding output layer with sigmoid activation
Classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the model
Classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.4,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and testing sets
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(64, 64),
                                                 batch_size=4,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(64, 64),
                                            batch_size=4,
                                            class_mode='binary')

# Train the model
Classifier.fit(training_set,
               steps_per_epoch=40,
               epochs=5,
               validation_data=test_set,
               validation_steps=8)

# Test the model with a new image
test_image_path = r'D:\new\Covid19-dataset\test\Covid'  # Replace with your actual test image file
test_image = image.load_img(test_image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict the result
result = Classifier.predict(test_image)

# Print prediction
if result[0][0] == 1:
    prediction = 'Normal'
    print(prediction)
else:
    prediction = 'COVID'
    print(prediction)

# Save the trained model
Classifier.save('model/covid_xray_model.h5')  # Fixed the model save method to use 'Classifier'
print("Model saved successfully as 'model/covid_xray_model.h5'")


#total there are 10 splits
