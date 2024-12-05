import os
from datetime import datetime
import pytz
import tensorflow as tf

def create_callbacks_from_model(model, parent_folder='movilstm', timezone='Asia/Shanghai'):
    """
    Create callbacks for the given model, preserving the original logic.

    Parameters:
        model (tf.keras.Model): The model whose name will be used in file paths.
        timezone (str): The timezone for timestamping model directory.

    Returns:
        list: List of callbacks (ModelCheckpoint and ReduceLROnPlateau).
    """
    # Define the timezones
    utc_tz = pytz.utc
    local_tz = pytz.timezone(timezone)

    # Convert current time to the target timezone
    local_time = datetime.now(utc_tz).astimezone(local_tz)

    # Ensure the parent folder exists
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Generate folder name using model name and timestamp
    model_name = model.name + '_' + local_time.strftime('%Y-%m-%d_%H-%M-%S') + '/'
    model_folder = os.path.join(parent_folder, model_name)

    # Create the folder if it does not exist
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # Print the created directory
    print(f"Directory created: {model_folder}")

    # Set the file path and naming convention for saved models
    filepath = os.path.join(
        model_folder,
        'model-{epoch:05d}-{loss:.5f}-{root_mean_squared_error:.5f}.h5'
    )

    # Create the ModelCheckpoint callback
    checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch'
    )

    # Create the ReduceLROnPlateau callback
    LR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        verbose=1,
        patience=4
    )

    # Return the callbacks list
    callbacks_list = [checkpoint1, LR]
    return callbacks_list, model_name