import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def show_bev (
    bev,
    channel=0,
    title="BEV Visualization",
    width=512,
    height=512,
    colormap='viridis'
) : 
    """
    Affiche une représentation BEV d'un canal spécifique.
    :param bev: Représentation BEV multi-canaux.
    :param channel: Canal à afficher (0: Intensité, 1: Hauteur, 2: Densité).
    :param title: Titre de la fenêtre de visualisation.
    :param width: Largeur de la fenêtre.
    :param height: Hauteur de la fenêtre.
    :param colormap: Colormap à utiliser pour l'affichage.
    """

    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(bev[channel], cmap=colormap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_lidar_point_cloud(file_path):
    """
    Lit un fichier LiDAR et affiche le nuage de points dans une fenêtre 3D.
    
    :param file_path: Chemin vers le fichier .bin contenant les données LiDAR.
    """
    # Lire les données LiDAR
    lidar_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    
    # Créer un nuage de points Open3D
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(lidar_data[:, :3])  # Utiliser uniquement x, y, z
    
    # Afficher le nuage de points
    o3d.visualization.draw_geometries([point_cloud])