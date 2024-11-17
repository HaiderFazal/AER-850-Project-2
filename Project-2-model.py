import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Data Processing

# Define the input image shape
input_shape = (500, 500, 3)
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 500, 500, 3
BATCH_SIZE = 32

train_dir = 'C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/train'
validation_dir = 'C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/valid'
test_dir = 'C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/test'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,              
    shear_range=0.2,              
    zoom_range=0.2,               
    horizontal_flip=True         
)

# Rescaling validation data
validation_datagen = ImageDataGenerator(rescale=1.0/255)        

# Create training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'      
)

# Create validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Step 2: Neural Network Architecture Design

# Define the number of classes (for example, 3 classes)
num_classes = 3

# Build the model
model = Sequential([
    # First convolutional layer with 32 filters, 3x3 kernel size, and 'same' padding
    Conv2D(32, (5, 5), activation='relu', input_shape=(500, 500, 3), padding='same'),
    MaxPooling2D(pool_size=(3, 3)),

    # Second convolutional layer with 64 filters, 3x3 kernel size
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(3, 3)),

    # Third convolutional layer with 128 filters, 3x3 kernel size
    Conv2D(128, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(3, 3)),

    # Flatten layer to convert 3D feature maps to 1D feature vectors
    Flatten(),

    # Fully connected dense layer with 128 neurons
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout for regularization

    # Final dense layer with 3 neurons (one for each class) and softmax activation
    Dense(num_classes, activation='softmax')
])

print("Model 1 Summary")
model.summary()

# Step 3: Hyperparamter Analysis

# Define hyperparameters
num_classes = 3  # Number of output classes
input_shape = (500, 500, 3)  # Input image shape
activation_conv = 'leaky_relu'  # Activation function for Conv2D layers
activation_dense = 'elu'  # Activation function for Dense layers
filters = [16, 32, 64]  # Number of filters for Conv2D layers
dense_units = 64  # Number of neurons in Dense layer
dropout_rate = 0.5  # Dropout rate
learning_rate = 0.001  # Learning rate for the optimizer

# Build the model
model = Sequential()

# Adding convolutional layers with hyperparameters
for filter_count in filters:
    model.add(Conv2D(filter_count, (3, 3), activation=None, input_shape=input_shape if len(model.layers) == 0 else None, padding='same'))
    if activation_conv == 'leaky_relu':
        model.add(tf.keras.layers.LeakyReLU())
    else:
        model.add(tf.keras.layers.Activation(activation_conv))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(dense_units, activation=None))
if activation_dense == 'elu':
    model.add(tf.keras.layers.Activation('elu'))
else:
    model.add(tf.keras.layers.Activation(activation_dense))

# Dropout for regularization
model.add(Dropout(dropout_rate))

# Output layer with softmax activation
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(optimizer=Adam(learning_rate=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
print()
print("Model 2 Hyperparameter Summary ")
model.summary()

# Step 4: Model Evaluation

history = model.fit(train_generator,epochs=25,validation_data=validation_generator)

# Save the history object
model.save("haider_model.h5")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()