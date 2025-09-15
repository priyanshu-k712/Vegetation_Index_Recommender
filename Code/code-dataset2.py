import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # Import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
# from google.colab import drive # Import for Google Drive mounting
from sklearn.metrics import r2_score # Import for R2 score calculation
import sys

# --- 0. Mount Google Drive (for Colab) ---
# This step is crucial to access your dataset stored in Google Drive.
# When you run this, a link will appear, click it, choose your Google account,
# copy the authorization code, and paste it back into the Colab cell.
# print("Mounting Google Drive...")
# try:
#     drive.mount('/content/drive')
#     print("Google Drive mounted successfully.")
# except Exception as e:
#     print(f"Error mounting Google Drive: {e}")
#     print("Please ensure you authorize Google Drive access when prompted.")

# --- Constants ---
DATASET_DIR = '/Dataset_Images/Multispectral_Images'  # Your dataset directory
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 150
INITIAL_LR = 0.001 # Used for ReduceLROnPlateau, though Adam typically manages its own LR

# --- 1. Data Loading Functions ---

def load_single_band_image(file_path):
    """
    Loads a single band image using OpenCV and resizes it.
    Assumes grayscale TIFF (cv2.IMREAD_UNCHANGED).

    Args:
        file_path (str): The full path to the image file.

    Returns:
        np.array: The loaded and resized 2D numpy array representing the image band.
                  Returns None if the file cannot be loaded or is invalid.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    # cv2.IMREAD_UNCHANGED ensures the image is loaded as is, preserving depth (e.g., 16-bit)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image from {file_path}")
        return None
    # Resize to a consistent size (IMG_WIDTH, IMG_HEIGHT)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return img_resized

def generate_and_load_dataset_from_files(dataset_dir):
    """
    Loads image data from the specified directory, groups by leaf sample ID,
    creates RGB inputs, and calculates corresponding vNDVI target values.

    Args:
        dataset_dir (str): The path to the directory containing the image files.

    Returns:
        tuple: A tuple containing:
            - np.array: X_data (RGB images), shape (num_samples, H, W, 3)
            - np.array: y_data (vNDVI values), shape (num_samples,)
    """
    X_data = [] # To store RGB images
    y_data = [] # To store vNDVI values

    print(f"Scanning directory: {dataset_dir}")
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}. Please check the path and if Drive is mounted.")
        return np.array([]), np.array([])

    # Group files by leaf sample ID
    leaf_samples = {}
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.tif'):
            # Expected format: leaf###dy_x.tif (e.g., leaf001d0_1.tif)
            parts = filename.split('_')
            if len(parts) == 2:
                leaf_id_full = parts[0] # e.g., 'leaf001d0'
                band_ext_part = parts[1] # e.g., '1.tif'

                try:
                    # Extract the three-digit leaf number (e.g., '001' from 'leaf001d0')
                    leaf_num_str = leaf_id_full[4:-2]
                    leaf_num = int(leaf_num_str)
                except (ValueError, IndexError):
                    print(f"Skipping malformed leaf ID in filename: {filename}")
                    continue

                try:
                    band_num = int(band_ext_part.split('.')[0]) # Extracts '1' from '1.tif'
                except ValueError:
                    print(f"Skipping malformed band number in filename: {filename}")
                    continue

                if leaf_num not in leaf_samples:
                    leaf_samples[leaf_num] = {}
                leaf_samples[leaf_num][band_num] = os.path.join(dataset_dir, filename)
            else:
                print(f"Skipping unexpected filename format: {filename}")

    print(f"Found {len(leaf_samples)} potential unique leaf samples.")

    # Process each leaf sample
    processed_count = 0
    for leaf_num in sorted(leaf_samples.keys()):
        sample_files = leaf_samples[leaf_num]

        # Check if all required bands are present for RGB and vNDVI calculation
        # RGB Input: Band 1 (Blue), Band 2 (Green), Band 3 (Red)
        # vNDVI Target: Band 1 (Blue), Band 2 (Green), Band 3 (Red)
        required_bands = {1, 2, 3} # Blue, Green, Red
        if not all(band in sample_files for band in required_bands):
            print(f"Skipping leaf {leaf_num:03d}: Missing one or more required bands (1, 2, 3).")
            continue

        # Load bands
        blue_band = load_single_band_image(sample_files[1])
        green_band = load_single_band_image(sample_files[2])
        red_band = load_single_band_image(sample_files[3])

        # Check if any band failed to load or resize
        if any(b is None for b in [blue_band, green_band, red_band]):
            print(f"Skipping leaf {leaf_num:03d}: Failed to load or resize one or more image files.")
            continue

        # --- Create RGB input image for the CNN ---
        # Stack as RGB (Red, Green, Blue) for common image display/CNN input order
        # Ensure data type is float32 and normalize pixel values to [0, 1]
        rgb_image = np.stack([red_band, green_band, blue_band], axis=-1).astype(np.float32) / 255.0
        X_data.append(rgb_image)

        # --- Calculate vNDVI target value ---
        # Ensure float type for calculation and normalize pixel values to [0, 1]
        blue_for_vndvi = blue_band.astype(np.float32) / 255.0
        green_for_vndvi = green_band.astype(np.float32) / 255.0
        red_for_vndvi = red_band.astype(np.float32) / 255

        # Add a small epsilon to avoid division by zero for negative exponents if value is zero
        epsilon = 1e-6
        red_for_vndvi = np.maximum(red_for_vndvi, epsilon)
        blue_for_vndvi = np.maximum(blue_for_vndvi, epsilon)

        # Calculate pixel-wise vNDVI using the provided formula
        # vNDVI = 0.5268 * (red^(-0.1294) * green^(0.3389) * blue^(-0.3118))
        pixel_vndvi = 0.5268 * (
            np.power(red_for_vndvi, -0.1294) *
            np.power(green_for_vndvi, 0.3389) *
            np.power(blue_for_vndvi, -0.3118)
        )

        # Average pixel-wise vNDVI to get a single value for the image
        avg_vndvi = np.mean(pixel_vndvi)
        y_data.append(avg_vndvi)
        processed_count += 1

        if processed_count % 50 == 0:
            print(f"Processed {processed_count} samples...")

    print(f"Finished loading. Successfully processed {processed_count} samples.")
    return np.array(X_data), np.array(y_data)

# --- Load the actual dataset ---
X, y = generate_and_load_dataset_from_files(DATASET_DIR)

# Check if data was loaded successfully
if X.size == 0 or y.size == 0:
    print("No data loaded. Exiting.")
    sys.exit() # Exit if no data is found, preventing further errors

# --- 2. Data Splitting ---
# Use the loaded data for splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # Using 0.2 as test_size for consistency
)

print(f"Training data shape (X_train): {X_train.shape}")
print(f"Training labels shape (y_train): {y_train.shape}")
print(f"Test data shape (X_test): {X_test.shape}")
print(f"Test labels shape (y_test): {y_test.shape}")

# --- 3. Build the Custom CNN Model ---
def build_custom_cnn(input_shape):
    """
    Builds a Custom CNN for image-level regression with increased depth and width,
    using a VGG-like structure and GlobalAveragePooling.
    Outputs a single vegetation index.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'), # Added third conv layer
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'), # Added third conv layer
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Global Average Pooling to reduce dimensions before dense layers
        layers.GlobalAveragePooling2D(),

        # Dense Layers
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # L2 regularization
        layers.Dropout(0.5), # Increased dropout
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # L2 regularization
        layers.Dropout(0.5), # Increased dropout

        # Output layer: single unit for regression (vNDVI prediction)
        # No activation function for regression tasks
        layers.Dense(1)
    ])

    # Compile the model
    # Using 'mse' (Mean Squared Error) as loss for regression
    # Using Adam optimizer with the initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# Get the input shape from our data
input_shape = X_train.shape[1:]
model = build_custom_cnn(input_shape)

# Print model summary
model.summary()

# --- 4. Define Callbacks ---
# EarlyStopping: Stop training when a monitored quantity has stopped improving.
early_stopping = EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=20,        # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Restore model vndvi_weights from the epoch with the best value of the monitored quantity
)

# ModelCheckpoint: Save the model after every epoch if validation loss improves.
model_checkpoint = ModelCheckpoint(
    filepath='best_model_vndvi.h5', # Path to save the model file (changed filename)
    monitor='val_loss',            # Monitor validation loss
    save_best_only=True,           # Only save when val_loss improves
    verbose=1
)

# ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # Monitor validation loss
    factor=0.5,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=10,        # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.00001,     # Lower bound on the learning rate
    verbose=1
)

callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

# --- 5. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2, # Further split training data for validation
    callbacks=callbacks_list, # Add the defined callbacks
    verbose=1
)
print("Model training complete.")

# --- 6. Evaluate the Model ---
print("\nEvaluating model on test data...")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(loss) # Calculate RMSE from MSE
y_pred_test = model.predict(X_test).flatten() # Make predictions once for R2 calculation
r2 = r2_score(y_test, y_pred_test) # Calculate R2 score

print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"Test MAE (Mean Absolute Error): {mae:.4f}")
print(f"Test R2 (Coefficient of Determination): {r2:.4f}")


# --- 7. Make Predictions and Visualize Results ---
print("\nMaking predictions on test data...")
# y_pred is already calculated from the R2 score calculation (y_pred_test)
y_pred = y_pred_test

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Actual vs. Predicted vNDVI
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction') # Diagonal line
plt.title('Actual vs. Predicted vNDVI') # Changed title
plt.xlabel('Actual vNDVI') # Changed label
plt.ylabel('Predicted vNDVI') # Changed label
plt.grid(True)
plt.legend()
plt.show()

# Display a few example images with their actual vs. predicted vNDVI
print("\nDisplaying example images with actual vs. predicted vNDVI:") # Changed text
import random # Ensure random is imported if not already
num_examples_to_show = min(5, len(X_test)) # Ensure we don't try to show more than available
indices = random.sample(range(len(X_test)), num_examples_to_show)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(indices):
    plt.subplot(1, num_examples_to_show, i + 1)
    # Display the RGB image (X_test is already normalized [0,1])
    plt.imshow(X_test[idx])
    plt.title(f"Actual: {y_test[idx]:.2f}\nPred: {y_pred[idx]:.2f}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\nProcess complete. Ensure your dataset is correctly placed in '/content/drive/MyDrive/Dataset_2'.")
