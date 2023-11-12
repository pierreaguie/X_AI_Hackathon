import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import imageio

import pandas as pd

#Si executé, ce fichier crée un nouveau dossier avec les images cropped classées en deux sous dossiers plume et no_plume.

def crop(path, filename, random = True ,ligne = None):
    """La fonction crop une image aléatoirement ou autour du centre de la plume.
    path: le path du dossier data
    filename: nom du fichier de l'image
    random: crop aléatoire ou non
    ligne: la ligne dans le fichier metadata.csv pour avoir le centre."""

    #ouverture de l'image
    image = Image.open(os.path.join(path,filename))
    image = np.array(image)
    
    if random:
        #selection aléatoire du carré à crop
        x = np.random.randint(16,64-16)
        y = np.random.randint(16,64-16)
    else:
        #carré contenant le centre de la plume
        x = np.clip(ligne['coord_x'], 16, 64-16)
        y = np.clip(ligne['coord_y'], 16, 64-16)

    x_min, x_max = x - 16, x + 16
    y_min, y_max = y - 16, y + 16

    #on met à 0 (noir) ce qui n'est pas crop
    for x in range(64):
        for y in range(64):
            if x < x_min or x >= x_max or y < y_min or y >= y_max:
                image[x,y] = 0
    return image

def crop_images_from_path(path):
    """
    La fonction parcours les fichiers du path et crop les images rencontrées
    Path : le chemin d'ccès au dossier.
    """
    images = []
    images_paths = []
    images_labels = []
    crops = []
    crops_paths = []
    crops_labels = []
    #on parcourt les différents dossiers du dossier principal
    for filename in os.listdir(path):
        if filename[-3:] != 'tif':
            continue
        img = Image.open(os.path.join(path,filename))
        img = np.array(img)
        
        if img is not None:
            
            #on charge les images
            img = Image.open(os.path.join(path,filename))
            img = np.array(img)
            # on accède à la ligne correspondante dans le metadata pour savoir s'il y a une plume et crop en conséquence
            ligne = metadata.loc[metadata['path'] == (path + '/' + filename)[:-4]]
            ligne = ligne.iloc[0]
            
            images.append(img)
            images_paths.append((path + '/' + filename)[:-4])
            images_labels.append(1 if ligne['plume']=='yes' else 0)
            
            if ligne['plume']=='yes'  :
                crops.append(crop(path, filename, random = False, ligne = ligne))
                crops_paths.append((path + '/' + filename)[:-4])
                crops_labels.append(1 if ligne['plume']=='yes' else 0)
                
            else :
                crops.append(crop(path, filename))
                crops_paths.append((path + '/' + filename)[:-4])
                crops_labels.append(1 if ligne['plume']=='yes' else 0)
                
    return [images, images_paths, images_labels], [crops, crops_paths, crops_labels]
    
def plot_images(images):

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image, cmap='gray')




metadata = pd.read_csv('metadata.csv')

path_p = 'images/plume'
images_p, crops_p = crop_images_from_path(path_p)
#images_p, crops_p = images_p[0], crops_p[0]

path_np = 'images/no_plume'
images_np, crops_np = crop_images_from_path(path_np)
#images_np, crops_np = images_np[0], crops_np[0]

#plot_images(images_p[:10])
#plot_images(crops_p[:10])

#plot_images(images_np[:10])
#plot_images(crops_np[:10])

#plt.show()

def save_images(images, paths):
    for i, image in enumerate(images):
        imageio.imwrite('crops/' + paths[i] + '.tif', image, format='TIFF')
    return

save_images(crops_np[0], crops_np[1])