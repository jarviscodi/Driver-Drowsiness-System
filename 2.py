import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Flatten
from tf.keras.optimizers import Adam

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Build an MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_flat.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_data=(X_test_flat, y_test))
