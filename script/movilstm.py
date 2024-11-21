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


def build_movilstm_model(backbone, material_name, heatflux_type):
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
        name=f'MoviLSTM_freeze_{material_name}_{heatflux_type}'
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model