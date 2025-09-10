import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Load the model
model = load_model('digit_recognizer_model.h5')
print("Model Loaded Successfully")


# Load the test data (We need raw data to make predictions)
(_,_),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

#Preprocess the data - Normalization and reshaping test data/images
X_test = X_test.astype('float32')/255.0
X_test = X_test.reshape(X_test.shape[0],28,28,1)

#Take a single image
img_to_predict = X_test[10]
actual_digit = y_test[10]

#since model expects a batch of images- so we need to add a dimension
img_to_predict_reshaped = np.expand_dims(img_to_predict,axis=0)

#make a prediction
prediction = model.predict(img_to_predict_reshaped)

# predicted digit 
predicted_digit = np.argmax(prediction)

#printing the result 
print("Predicted Digit : ",predicted_digit)
print("Actual Digit : ",actual_digit)

# model.summary()

plt.imshow(img_to_predict.reshape(28,28),cmap='gray')
plt.title(f'Predicted_digit:{predicted_digit} , Actual_digit:{actual_digit}')
plt.axis('off')
plt.show()