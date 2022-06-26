# Import stuff
# The models
from camperock.u_net_3_channels import UNet
from camperock.u_net_exp_3_channels import ExpandedUNet
from camperock.u_net_res_3_channels import ResUNet
from camperock.u_net_inception_3_channels import InceptionUNet

from camperock.metrics import iou, true_positive, false_positive, dice_coeff
from camperock.generate_files import generate_results_file, generate_training_and_testing_files
from camperock.functions import elastic_transform, shift_image

import pandas as pd

from tqdm.keras import TqdmCallback

import numpy as np
#np.random.seed(69)
import cv2
import tensorflow as tf
#tf.random.set_seed(69)

import random
import matplotlib.pyplot as plt
import os

# Define models, all with the same number of filters
unet_filters = [16, 32, 64, 128, 256]
models = [
    #UNet(unet_filters, (5,5), 'same', 1), # U-Net with 4 outputs
    ExpandedUNet(unet_filters, (5,5), 'same', 1), # U-Net with 4 decoding paths
    #ResUNet(unet_filters, (5,5), 'same', 1, with_dense = False), # Normal ResUNet
    #ResUNet(unet_filters, (5,5), 'same', 1, with_dense = True), # U-Net with Dense path
    #InceptionUNet(unet_filters, (5,5), 'same', 1) # Inception U-Net
]
model_names = [
    #'u_net',
    'expanded_u_net',
    #'res_u_net',
    #'res_dense_u_net',
    #'inception_u_net'
]

images_folder = 'bloque_1_y_3_color'

label_types = ['RG', 'RB', 'GB', 'RGB']
image_size = 256
image_names = os.listdir(f'images/{images_folder}')
# For replicability purposes
random.seed(69)
random.shuffle(image_names)

#-------------------------------------------
# Super ultra crazy training benchmark loop
#-------------------------------------------
# FOR 5-FOLD CROSS-VALIDATION

def load_mask(image_name):
    mask = np.zeros((image_size, image_size, 4))
    # Load the label for each channel combination for the image
    for i in range(len(label_types)):
        mask_temp = plt.imread(f'labels/{label_types[i]}_labels/{image_name}')
        mask_temp = cv2.resize(mask_temp, (image_size, image_size))
        mask[..., i] = mask_temp
    return mask


# Load and prepare training images
def load_training_images_and_labels(X_train_names):
    images = np.zeros((0, image_size, image_size, 3))

    # Final mask list
    masks = np.zeros((0, image_size, image_size, 4))
    image_counter = 0
    #for image_name in image_names:
    for image_name in X_train_names:
        # Load and resize 3-channel images
        image = plt.imread(f'images/{images_folder}/{image_name}')
        image = cv2.resize(image, (image_size, image_size))
        
        # Populate final arrays
        #--------------------------------
        # Original image and mask
        #--------------------------------
        images = np.append(images, np.expand_dims(image, 0), axis = 0)
        mask = load_mask(image_name)
        masks = np.append(masks, np.expand_dims(mask, 0), axis = 0)
        
        #--------------------------------
        # Rotated image and mask
        #--------------------------------
        rot_times = np.random.randint(1, 4)
        rotated_image = np.rot90(image.copy(), k = rot_times)
        rotated_mask = np.rot90(mask.copy(), k = rot_times)
        
        
        images = np.append(
            images,
            np.expand_dims(
                rotated_image.copy(),0
            ), axis = 0
        )
        
        masks = np.append(
            masks,
            np.expand_dims(
                rotated_mask.copy(), 0
            ), axis = 0
        )
        
        #----------------------------------
        # Random elastic deformations
        #----------------------------------
        # Distort original image and mask
        random_seed = np.random.randint(0, 1000)
        distorted_image = elastic_transform(image.copy(), random_state = random_seed)
        distorted_mask = elastic_transform(mask.copy(), random_state = random_seed)
        # Distort rotate image and mask
        random_seed = np.random.randint(0, 1000)
        distorted_rot_image = elastic_transform(rotated_image.copy(), random_state = random_seed)
        distorted_rot_mask = elastic_transform(rotated_mask.copy(), random_state = random_seed)
        
        images = np.append(images, np.expand_dims(distorted_image, 0), axis = 0)
        images = np.append(images, np.expand_dims(distorted_rot_image, 0), axis = 0)
        masks = np.append(masks, np.expand_dims(distorted_mask, 0), axis = 0)
        masks = np.append(masks, np.expand_dims(distorted_rot_mask, 0), axis = 0)
        
        
    # Make masks be between 0 and 1
    masks = masks / 255
    images = images / 255
    return images, masks


for model, model_name in zip(models, model_names):
    final_df = {
        'model_name': [],
        'k_fold': [],
        'metric_name': [],
        'output_number': [],
        'metric': []
    }

    print('\n')
    print('----------------------------------------')
    print(f'Starting model {model_name}')
    print('----------------------------------------')
    print('\n')

    # Same optimizer for everyone
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    #Compile the model
    model.compile(optimizer = optimizer,
        metrics = [iou, true_positive, false_positive, dice_coeff],
        loss = 'binary_crossentropy' 
    )

    # Save initial weights for use in each k-fold
    model.save_weights(f'u_net_results/k_fold_weights/{model_name}/initial/{model_name}')

    # Run the five folds
    for k in range(1):
        print(f'Current k: {k}')
        # Select the images for the current run
        val_index_k = k
        X_val_names = [image_names[i * 11 : (i+1) * 11] for i in range(5)]
        X_train_names = [name for name in image_names if name not in X_val_names[val_index_k]]

        images, masks = load_training_images_and_labels(X_train_names)

        #-----------------------------------
        # Load validation stuff
        #-----------------------------------
        X_val = np.zeros((0, image_size, image_size, 3))

        # Final mask list
        y_val = np.zeros((0, image_size, image_size, 4))

        # Procedure for the remaining images
        #for image_name in X_val_names:
        for image_name in X_val_names[val_index_k]:
            # Load and resize 3-channel images
            image = plt.imread(f'images/{images_folder}/{image_name}')
            image = cv2.resize(image, (image_size, image_size))
            
            # Populate final arrays
            #--------------------------------
            # Original image and mask
            #--------------------------------
            X_val = np.append(X_val, np.expand_dims(image, 0), axis = 0)
            mask = load_mask(image_name)
            y_val = np.append(y_val, np.expand_dims(mask, 0), axis = 0)

        X_val = X_val / 255
        y_val = y_val / 255


        #-----------------------------------
        # Train the model
        #-----------------------------------
        history = model.fit(
            images,
            [masks[...,0], masks[...,1], masks[...,2], masks[...,3]],
            epochs = 20,
            batch_size = 4,
            verbose = 0,
            callbacks = [TqdmCallback(verbose=1)]
        )

        evaluation = model.evaluate(X_val, [y_val[...,0], y_val[...,1], y_val[...,2], y_val[...,3]])
        

        for i in range(16):
            final_df['model_name'].append(model_name)
            final_df['k_fold'].append(k)

        # First output
        final_df['metric_name'].append('iou')
        final_df['output_number'].append(1)
        final_df['metric'].append(evaluation[5])
        final_df['metric_name'].append('true_positive')
        final_df['output_number'].append(1)
        final_df['metric'].append(evaluation[6])
        final_df['metric_name'].append('false_positive')
        final_df['output_number'].append(1)
        final_df['metric'].append(evaluation[7])
        final_df['metric_name'].append('dice_coeff')
        final_df['output_number'].append(1)
        final_df['metric'].append(evaluation[8])

        # Second output
        final_df['metric_name'].append('iou')
        final_df['output_number'].append(2)
        final_df['metric'].append(evaluation[9])
        final_df['metric_name'].append('true_positive')
        final_df['output_number'].append(2)
        final_df['metric'].append(evaluation[10])
        final_df['metric_name'].append('false_positive')
        final_df['output_number'].append(2)
        final_df['metric'].append(evaluation[11])
        final_df['metric_name'].append('dice_coeff')
        final_df['output_number'].append(2)
        final_df['metric'].append(evaluation[12])

        # Third output
        final_df['metric_name'].append('iou')
        final_df['output_number'].append(3)
        final_df['metric'].append(evaluation[13])
        final_df['metric_name'].append('true_positive')
        final_df['output_number'].append(3)
        final_df['metric'].append(evaluation[14])
        final_df['metric_name'].append('false_positive')
        final_df['output_number'].append(3)
        final_df['metric'].append(evaluation[15])
        final_df['metric_name'].append('dice_coeff')
        final_df['output_number'].append(3)
        final_df['metric'].append(evaluation[16])

        # Fourth output
        final_df['metric_name'].append('iou')
        final_df['output_number'].append(4)
        final_df['metric'].append(evaluation[17])
        final_df['metric_name'].append('true_positive')
        final_df['output_number'].append(4)
        final_df['metric'].append(evaluation[18])
        final_df['metric_name'].append('false_positive')
        final_df['output_number'].append(4)
        final_df['metric'].append(evaluation[19])
        final_df['metric_name'].append('dice_coeff')
        final_df['output_number'].append(4)
        final_df['metric'].append(evaluation[20])

        model.save_weights(f'u_net_results/k_fold_weights/{model_name}/k_{k}/{model_name}')
        # Restore original weights
        #model.load_weights(f'u_net_results/k_fold_weights/{model_name}/initial/{model_name}')

    final_df = pd.DataFrame(final_df)
    final_df.to_excel(f'benchmark_results/{model_name}.xlsx', index = False)