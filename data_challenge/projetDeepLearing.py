import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Define some hyperparameters
img_height = 48
img_width = 48
batch_size = 32
num_classes = 2
epochs = 50

#repoertories for images
train_dir = './train'
test_dir = './test'


int a = 3

#######################################################################
###################### Train ###########################################

# load images
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Load images
train_images = load_images_from_folder(train_dir)
test_images = load_images_from_folder(test_dir)

# Load labels and genders
image_names, labels, genders = np.loadtxt("train.txt", dtype=str, unpack=True)

# rescale
train_images = [img.astype('float32') / 255.0 for img in train_images]

# Flatten images
train_images_flat = [img.flatten() for img in train_images]


# Split data
X_train, X_val, y_train, y_val, gender_train, gender_val = train_test_split(
    train_images_flat, labels, genders, test_size=0.2
)


# Convert labels to categorical
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Use ImageDataGenerator to load and preprocess images 
# train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)


# Train the model
#history = model.fit(train_generator, epochs=epochs, validation_data=(X_val, y_val))
model.fit(X_train, y_train)

#Make predictions on the validation set
predictions = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, predictions)
print(f"Accuracy: {accuracy}")

#######################################################################
###################### Test ###########################################

# Make predictions on the test set
test_folder_path = "/test"

test_image_names, test_labels, test_genders = np.loadtxt("train.txt", dtype=str, unpack=True)

# rescale
test_images = [img.astype('float32') / 255.0 for img in test_images]

# Flatten images
test_images_flat = [img.flatten() for img in test_images]

test_predictions = model.predict(test_images_flat)

# Generate output file
output_file = "predictions.txt"
with open(output_file, 'w') as f:
    for img_name, prediction in zip(test_image_names, test_predictions):
        f.write(f"{img_name} {prediction}\n")

