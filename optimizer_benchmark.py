from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop, Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import time

# Load and preprocess CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode class labels
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Initialize a sequential CNN model
conv2D_model: Sequential = Sequential()

# The number of filters for each convolutional layer is based on the reversed student ID (333814994)
# This personalized design helps create a consistent but unique architecture across experiments

# Convolutional Block 1
conv2D_model.add(Conv2D(81, (3, 3), strides=(2, 2), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False,
                        input_shape=(32, 32, 3)))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 2
conv2D_model.add(Conv2D(94, (3, 3), strides=(2, 2), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 3
conv2D_model.add(Conv2D(99, (3, 3), strides=(2, 2), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 4
conv2D_model.add(Conv2D(49, (3, 3), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 5
conv2D_model.add(Conv2D(14, (3, 3), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 6
conv2D_model.add(Conv2D(81, (3, 3), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 7
conv2D_model.add(Conv2D(38, (3, 3), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Convolutional Block 8
conv2D_model.add(Conv2D(33, (3, 3), padding='same',
                        activation='relu', kernel_regularizer=l2(0.001),
                        kernel_initializer='he_normal', use_bias=False))
conv2D_model.add(BatchNormalization())
conv2D_model.add(Dropout(0.3))

# Flatten the 2D feature maps to a 1D feature vector
conv2D_model.add(Flatten())

# Fully connected layer
conv2D_model.add(Dense(33, activation='relu', kernel_regularizer=l2(0.001)))
conv2D_model.add(Dropout(0.3))

# Output layer for 10-class classification (CIFAR-10)
conv2D_model.add(Dense(10, activation='softmax'))

# Print the model summary for architectural overview
conv2D_model.summary()

# Compile the model using NAG optimizer (SGD with momentum and nesterov=True)
conv2D_model.compile(
    optimizer=Adadelta(learning_rate=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Record the training start time
start_time = time.time()

# Train the model on 80% of the data (20% used for validation)
training = conv2D_model.fit(
    X_train, Y_train,
    epochs=65,
    batch_size=64,
    validation_split=0.2
)

# Calculate total training time
end_time = time.time()
training_time = end_time - start_time

# Evaluate model on test dataset
test_loss, test_acc = conv2D_model.evaluate(X_test, Y_test, verbose=2)

# Predict class probabilities on test data
Y_test_pred = conv2D_model.predict(X_test)
Y_pred_classes = np.argmax(Y_test_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Compute evaluation metrics
precision = precision_score(Y_true, Y_pred_classes, average='macro')
recall = recall_score(Y_true, Y_pred_classes, average='macro')
f1 = f1_score(Y_true, Y_pred_classes, average='macro')

# Print results
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")
print(f"Training time: {training_time / 60:.2f} minutes")
