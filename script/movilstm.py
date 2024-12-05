import tensorflow as tf
from official.projects.movinet.modeling import movinet # type: ignore
from official.projects.movinet.modeling import movinet_model # type: ignore

def build_and_load_backbone():
    """
    Builds MoviNet models (trainable, frozen, and initial), and loads pretrained weights.
    The script includes:
        - Creating three MoviNet models: trainable, frozen, and initial.
        - Loading pretrained weights for transfer learning.
        - Setting the trainable status for layers in each model.
    """
    # Model parameters
    num_classes = 600
    model_id = 'a0'
    checkpoint_dir = 'movinet_a0_base'

    # Clear the previous session
    tf.keras.backend.clear_session()

    # Initialize three backbone models
    backbone = movinet.Movinet(model_id=model_id)
    backbone_freeze = movinet.Movinet(model_id=model_id)
    backbone_initial = movinet.Movinet(model_id=model_id)

    # Build MoviNet classifiers
    pretrained_model_1 = movinet_model.MovinetClassifier(backbone=backbone, num_classes=num_classes)
    pretrained_model_1.build([1, 1, 1, 1, 3])

    pretrained_model_2 = movinet_model.MovinetClassifier(backbone=backbone_freeze, num_classes=num_classes)
    pretrained_model_2.build([1, 1, 1, 1, 3])

    # Load pretrained weights
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in directory: {checkpoint_dir}")

    checkpoint1 = tf.train.Checkpoint(model=pretrained_model_1)
    status1 = checkpoint1.restore(checkpoint_path)
    status1.assert_existing_objects_matched()

    checkpoint2 = tf.train.Checkpoint(model=pretrained_model_2)
    status2 = checkpoint2.restore(checkpoint_path)
    status2.assert_existing_objects_matched()

    print(f"Pretrained weights successfully loaded from {checkpoint_path}")

    # Set layer trainable status
    for layer in backbone.layers:
        layer.trainable = True
    print("Backbone layers 1 are set to trainable.")

    for layer in backbone_freeze.layers:
        layer.trainable = False
    print("Backbone layers 2 are frozen.")

    for layer in backbone_initial.layers:
        layer.trainable = True
    print("Initial backbone layers are set to trainable.")

    print("MoviNet models are ready to use!")

    return backbone, backbone_freeze, backbone_initial, pretrained_model_1, pretrained_model_2, checkpoint_path


def build_movilstm_model(backbone, material_name, heatflux_type, case_number):
    """
    Build the MoviLSTM model based on a frozen backbone model.

    Parameters:
        backbone_freeze (tf.keras.Model): The pre-trained backbone with frozen layers.
        material_name (str): Name of the material (for naming purposes).
        heatflux_type (str): Type of heat flux (for naming purposes).

    Returns:
        tf.keras.Model: The compiled MoviLSTM model.
    """
    # Extract output from the backbone
    backbone_output = backbone.outputs[10]
    
    # ConvLSTM2D layers to process the sequence
    x = tf.keras.layers.ConvLSTM2D(
        filters=64, kernel_size=(3, 3),
        padding='same', activation='relu', return_sequences=True
    )(backbone_output)
    
    # Conv3DTranspose layers to upsample the spatial dimensions
    x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same')(x)
    x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=(1, 3, 2), padding='same')(x)
    x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=(1, 5, 2), padding='same')(x)
    x = tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same')(x)
    
    # Final ConvLSTM2D layers to match the desired output dimensions
    x = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    decoder_outputs = tf.keras.layers.ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    
    # Define the model
    model = tf.keras.Model(
        inputs=backbone.inputs[0],
        outputs=decoder_outputs,
        name=f'MoviLSTM_freeze_{material_name}_{heatflux_type}_{case_number}'
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def build_convlstm_model(input_shape, material_name, heatflux_type, case_number):
    """
    Build a ConvLSTM model for spatiotemporal data processing.

    Parameters:
        input_shape (tuple): Shape of the input tensor (time_steps, height, width, channels).
        model_name (str): Name of the model.

    Returns:
        tf.keras.Model: The compiled ConvLSTM model.
    """
    # Define the input
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')

    # Add ConvLSTM2D layers
    x = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(inputs)
    x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)

    # Final output layer
    outputs = tf.keras.layers.ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)

    # Create the model
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name=f'ConvLSTM_{material_name}_{heatflux_type}_{case_number}'
        )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def build_convlstm_model_9(input_shape, material_name, heatflux_type, case_number):
    """
    Build a ConvLSTM model for spatiotemporal data processing.

    Parameters:
        input_shape (tuple): Shape of the input tensor (time_steps, height, width, channels).
        model_name (str): Name of the model.

    Returns:
        tf.keras.Model: The compiled ConvLSTM model.
    """
    # Define the input
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')

    # Add ConvLSTM2D layers
    x = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(inputs)
    x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)

    # Final output layer
    outputs = tf.keras.layers.ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True)(x)

    # Create the model
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name=f'ConvLSTM_{material_name}_{heatflux_type}_{case_number}'
        )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def build_c3d_convlstm2d(input_shape, 
                        filters=[16, 32, 64, 32, 16, 3], 
                        kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3), (3, 3), (3, 3), (3, 3)], 
                        padding='same', 
                        activation='relu', 
                        optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['mean_squared_error']):
    """
    Build a Conv3D + ConvLSTM2D model.

    Args:
        input_shape (tuple): Input shape (time_steps, height, width, channels).
        filters (list): List of filters for Conv3D and ConvLSTM2D layers.
        kernel_sizes (list): List of kernel sizes for Conv3D and ConvLSTM2D layers.
        padding (str): Padding for all layers (default is 'same').
        activation (str): Activation function for all layers (default is 'relu').
        optimizer (str): Optimizer for model compilation.
        loss (str): Loss function for model compilation.
        metrics (list): List of metrics for model compilation.

    Returns:
        tf.keras.Model: Compiled TensorFlow model.
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Conv3D layer
    x = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=kernel_sizes[0], 
                               padding=padding, activation=activation)(inputs)
    
    # Additional Conv3D layer
    x = tf.keras.layers.Conv3D(filters=filters[1], kernel_size=kernel_sizes[1], 
                               padding=padding, activation=activation)(x)
    
    # ConvLSTM2D layers
    x = tf.keras.layers.ConvLSTM2D(filters=filters[2], kernel_size=kernel_sizes[2], 
                                   padding=padding, activation=activation, return_sequences=True)(x)
    
    x = tf.keras.layers.ConvLSTM2D(filters=filters[3], kernel_size=kernel_sizes[3], 
                                   padding=padding, activation=activation, return_sequences=True)(x)
    
    x = tf.keras.layers.ConvLSTM2D(filters=filters[4], kernel_size=kernel_sizes[4], 
                                   padding=padding, activation=activation, return_sequences=True)(x)
    
    # Final ConvLSTM2D layer
    outputs = tf.keras.layers.ConvLSTM2D(filters=filters[5], kernel_size=kernel_sizes[5], 
                                         padding=padding, activation=activation, return_sequences=True)(x)
    
    # Model definition
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='C3DConvLSTM2D')
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.get(optimizer), 
                  loss=loss, 
                  metrics=metrics)
    
    return model