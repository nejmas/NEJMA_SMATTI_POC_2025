import numpy as np
import tensorflow as tf

import os

from App.config import WIDTH, HEIGHT, NUMBER_OF_CHANNELS, Z_SCALE, RANDOM_MIN_VALUE, RANDOM_MAX_VALUE

def read_kitti_lidar(file_path):
    """
    Lit les données LiDAR du dataset KITTI.
    :param file_path: Chemin vers le fichier .bin contenant les données LiDAR.
    :return: Nuage de points sous forme de tableau numpy (N, 4) [x, y, z, intensité].
    """
    lidar_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return lidar_data

def lidar_to_bev(lidar_data, bev_height=512, bev_width=512, z_scale=1.0):
    """
    Transforme les données LiDAR en représentation BEV multi-canaux.
    :param lidar_data: Tableau numpy contenant les données LiDAR (N, 4).
    :param bev_height: Hauteur de la matrice BEV.
    :param bev_width: Largeur de la matrice BEV.
    :param z_scale: Échelle pour la hauteur z.
    :return: Représentation BEV multi-canaux.
    """
    bev = np.zeros((3, bev_height, bev_width), dtype=np.float32)  # Multi-canaux : [intensité, hauteur, densité]
    x = lidar_data[:, 0]
    y = lidar_data[:, 1]
    z = lidar_data[:, 2]
    intensity = lidar_data[:, 3]

    # Transformation des coordonnées en indices BEV
    x_indices = np.int32((x - np.min(x)) / (np.max(x) - np.min(x)) * (bev_width - 1))
    y_indices = np.int32((y - np.min(y)) / (np.max(y) - np.min(y)) * (bev_height - 1))

    # Canal 1 : Intensité
    bev[0, y_indices, x_indices] = intensity

    # Canal 2 : Hauteur (z)
    bev[1, y_indices, x_indices] = z * z_scale

    # Canal 3 : Densité
    for i in range(len(x_indices)):
        bev[2, y_indices[i], x_indices[i]] += 1

    bev[2] = np.clip(bev[2], 0, 1)  # Normalisation de la densité entre 0 et 1
    return bev

def replace_zeros_with_random(bev, min_value=0.01, max_value=0.05):
    """
    Remplace les zéros dans les matrices BEV par des petites valeurs aléatoires.
    :param bev: Représentation BEV multi-canaux.
    :param min_value: Valeur minimale aléatoire.
    :param max_value: Valeur maximale aléatoire.
    :return: BEV avec les zéros remplacés.
    """
    random_values = np.random.uniform(min_value, max_value, bev.shape)
    bev[bev == 0] = random_values[bev == 0]
    return bev

def create_bev_dataset_from_folder(folder_path, batch_size=32):
    """
    Crée un dataset TensorFlow à partir des fichiers LiDAR dans un dossier.
    
    :param folder_path: Chemin vers le dossier contenant les fichiers .bin LiDAR.
    :param batch_size: Taille des batches pour le dataset.
    :return: Dataset TensorFlow prêt pour l'entraînement.
    """
    def parse_lidar_file(file_path):
        """
        Lit un fichier LiDAR, le transforme en BEV multi-canaux.
        """
        lidar_data = read_kitti_lidar(file_path.numpy().decode('utf-8'))
        bev = lidar_to_bev(
            lidar_data=lidar_data,
            bev_height=HEIGHT,
            bev_width=WIDTH,
            z_scale=Z_SCALE
        )
        bev_with_random = replace_zeros_with_random(
            bev=bev,
            min_value=RANDOM_MIN_VALUE,
            max_value=RANDOM_MAX_VALUE
        )
        # Réorganiser les axes pour obtenir [height, width, channels]
        bev_with_random = np.transpose(bev_with_random, (1, 2, 0))
        return bev_with_random

    def tf_parse_lidar_file(file_path):
        """
        Wrapper pour appeler parse_lidar_file avec tf.py_function.
        """
        bev = tf.py_function(parse_lidar_file, [file_path], Tout=tf.float32)
        bev.set_shape((HEIGHT, WIDTH, NUMBER_OF_CHANNELS))  # Fixer la forme
        return bev, bev  # Entrée et cible identiques (pour un autoencodeur)

    # Récupérer tous les fichiers .bin dans le dossier
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bin')]

    # Créer un dataset TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(tf_parse_lidar_file, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Performance optimizations
    dataset = dataset.cache()  # Mise en cache des données

    return dataset