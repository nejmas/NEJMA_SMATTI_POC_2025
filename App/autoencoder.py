import numpy as np
import tensorflow as tf

from App.config import INPUT_SHAPE, BOTTLENECK_NUMBER_OF_FEATURE, NUMBER_OF_CHANNELS

def create_bev_encoder(input_shape, bottleneck):
    """
    Crée un modèle fonctionnel pour encoder les BEV.
    
    :param input_shape: Tuple représentant la forme de l'entrée (hauteur, largeur, canaux).
    :param bottleneck: Nombre de neurones dans la couche de goulot d'étranglement.
    :return: Modèle Keras fonctionnel.
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")

    # Première couche convolutionnelle
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name="pool1")(x)
    x = tf.keras.layers.BatchNormalization(name="conv1_bn")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="conv1_act")(x)

    # Deuxième couche convolutionnelle
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name="pool2")(x)
    x = tf.keras.layers.BatchNormalization(name="conv2_bn")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="conv2_act")(x)

    # Troisième couche convolutionnelle
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name="conv3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name="pool3")(x)
    x = tf.keras.layers.BatchNormalization(name="conv3_bn")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="conv3_act")(x)

    # Aplatissement et couche dense (bottleneck)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="convFlatten_act")(x)
    outputs = tf.keras.layers.Dense(bottleneck, name="bottleneck_layer")(x)

    # Création du modèle
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BevEncoder")
    return model

def create_bev_decoder(bottleneck):
    """
    Crée un modèle fonctionnel pour encoder les BEV.
    
    :param input_shape: Tuple représentant la forme de l'entrée (hauteur, largeur, canaux).
    :param bottleneck: Nombre de neurones dans la couche de goulot d'étranglement.
    :return: Modèle Keras fonctionnel.
    """

    inputs = tf.keras.layers.Input(shape=(bottleneck,), name="input_layer")

    # Dense layer for reshaping
    x = tf.keras.layers.Dense(16 * 16 * 128, activation='relu', name="dense_layer")(inputs)
    x = tf.keras.layers.Reshape((16, 16, 128), name="reshape_layer")(x)

    # Première couche de convolution transposée
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name="deconv1")(x)
    x = tf.keras.layers.BatchNormalization(name="deconv1_bn")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="deconv1_act")(x)

    # Deuxième couche de convolution transposée
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name="deconv2")(x)
    x = tf.keras.layers.BatchNormalization(name="deconv2_bn")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="deconv2_act")(x)

    # Troisième couche de convolution transposée
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name="deconv3")(x)
    x = tf.keras.layers.BatchNormalization(name="deconv3_bn")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name="deconv3_act")(x)

    # Couche de sortie
    outputs = tf.keras.layers.Conv2DTranspose(NUMBER_OF_CHANNELS, (3, 3), activation='sigmoid', padding='same', name="output_layer")(x)

    # Création du modèle
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BevDecoder")

    return model

# Create the AutoEncoder model by combining the encoder and decoder

def create_bev_autoencoder(input_shape, bottleneck):
    """
    Crée un modèle AutoEncoder pour les BEV.
    
    :param input_shape: Tuple représentant la forme de l'entrée (hauteur, largeur, canaux).
    :param bottleneck: Nombre de neurones dans la couche de goulot d'étranglement.
    :return: Modèle Keras fonctionnel.
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    
    # Encodeur
    encoder = create_bev_encoder(input_shape, bottleneck)
    encoded = encoder(inputs)

    # Décodeur
    decoder = create_bev_decoder(bottleneck)
    decoded = decoder(encoded)

    # Création du modèle AutoEncoder
    autoencoder = tf.keras.Model(inputs=inputs, outputs=decoded, name="BevAutoencoder")
    
    return autoencoder
