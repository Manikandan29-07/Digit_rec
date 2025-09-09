import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, Input
# Load the MNIST dataset
(X_train, y_train),(X_test, y_test) = mnist.load_data()
# print(f'Training data shape: {X_train.shape}')
# print(f'Training labels shape: {y_train.shape}')
# print(f'Test data shape: {X_test.shape}')
# print(f'Test labels shape: {y_test.shape}')

# Data preprocessing 

# Normalization: The pixel values of the images are between 0 and 255. Normalizing them to a range of 0 to 1 helps the neural network converge faster.
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshaping: A Convolutional Neural Network (CNN) expects a specific input shape. We need to reshape the images to (batch_size, height, width, channels). For our grayscale images, the number of channels is 1.
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

# Dimensions after resizing the images
# print("Reshaped Training data Shape : ",X_train.shape)
# print("Reshaped Training data Shape : ",X_test.shape)


# Defining the model architecture
model = models.Sequential([
    Input(shape = (28, 28, 1)),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    
    layers.Flatten(),                                # Converts the 2D feature maps into a 1D vector so that it can be feed to fully connected layers
    layers.Dense(64, activation = 'relu'),          # 64 neurons with ReLU → learns complex combinations of features.
    layers.Dense(10, activation = 'softmax')        # 10 neurons with Softmax → probabibility distribution over 10 classes (digits 0 - 9 in MNIST).
    ]
)

# Model summary
# model.summary()


# Compile the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


# Train the model 
history = model.fit(
    X_train,
    y_train,
    epochs = 6,
    batch_size = 32,
    validation_data = (X_test,y_test)
)

print("Model training is complete")

# Save the trained model to a file
model.save('digit_recognizer_model.h5')   # Next time store as .keras file