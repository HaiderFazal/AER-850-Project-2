import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Data Processing

img_layout= (500, 500, 3) # image width, height, channel
BATCH_SIZE = 32

train_dir = 'C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/train'
validation_dir = 'C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/valid'
test_dir = 'C:/Users/Saira/Desktop/Uni/Year 4/Sem 1/AER-850/Project-2/Project 2 Data/Data/test'

train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.0/255)        

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(500, 500), batch_size=32,class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(500, 500),batch_size=32,class_mode='categorical')

# Step 2: Neural Network Architecture Design

num_classes = 3 # num of classes

model = Sequential([

    Conv2D(32, (5, 5), activation='relu', input_shape=(500, 500, 3), padding='same'), # 1st Conv layer
    MaxPooling2D(pool_size=(3, 3)),

    Conv2D(64, (5, 5), activation='relu', padding='same'), # 2nd Conv Layer 
    MaxPooling2D(pool_size=(3, 3)),

    Conv2D(128, (5, 5), activation='relu', padding='same'), # 3rd Conv Layer
    MaxPooling2D(pool_size=(3, 3)),

    Flatten(),

    Dense(128, activation='relu'), # 128 neurons
    Dropout(0.5), 

    Dense(num_classes, activation='softmax')
])

print("Model 1 Summary")
model.summary()

# Step 3: Hyperparamter Analysis

num_classes = 3 
input_shape = (500, 500, 3) 
activation_conv = 'leaky_relu'
activation_dense = 'elu' 
filters = [16, 32, 64] # can be changed in power of base 2^x
dense_units = 64 
dropout_rate = 0.5 
learning_rate = 0.001 

model = Sequential()

for filter_count in filters:
    model.add(Conv2D(filter_count, (3, 3), activation=None, input_shape=input_shape if len(model.layers) == 0 else None, padding='same'))
    if activation_conv == 'leaky_relu':
        model.add(tf.keras.layers.LeakyReLU())
    else:
        model.add(tf.keras.layers.Activation(activation_conv))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(dense_units, activation=None))
if activation_dense == 'elu':
    model.add(tf.keras.layers.Activation('elu'))
else:
    model.add(tf.keras.layers.Activation(activation_dense))

model.add(Dropout(dropout_rate))

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

print()
print("Model 2 Hyperparameter Summary ")
model.summary()

# Step 4: Model Evaluation

history = model.fit(train_generator,epochs=25,validation_data=validation_generator)

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