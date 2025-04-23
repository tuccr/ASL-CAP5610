import os
import random
import gc
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

def load_data(images_path: str):
    image_folders = os.listdir(images_path)

    # Pre-allocate image_data to ensure memory is reserved
    SKIP_VAL = 1
    NUM_CLASSES= 28
    image_data= np.empty(((3000//SKIP_VAL)*NUM_CLASSES*6, 128, 128, 1), dtype=np.float16)
    print("RAM used to store image data: " + str(sys.getsizeof(image_data) // (1024*1024)) + " MB")
    labels = []
    index = 0  # This will keep track of where you are in the array

    for i in range(len(image_folders)):  # only use every five images from raw dataset
        gc.collect()
        print("Now loading in images from folder: " + str(image_folders[i]) + ", time: " + str(datetime.datetime.now().hour) + ":" + str(datetime.datetime.now().minute))

        # open up a folder corresponding to the hand gesture
        letter_folder_path = os.path.join(images_path, image_folders[i])
        folder_imgs = os.listdir(letter_folder_path)

        # open up each image in the subfolder and save data/labels
        for img in range(0, len(folder_imgs), SKIP_VAL):
            # read raw image
            image_path = os.path.join(letter_folder_path, folder_imgs[img])
            temp_data = cv2.imread(image_path)

            # resize to 128x128
            mod_img = cv2.resize(temp_data, (128, 128), cv2.INTER_LINEAR)

            # apply brightness and contrast, convert to grayscale
            brightness = random.randint(-50, 50)
            contrast = random.uniform(0.8, 1.2)
            mod_img = mod_img.astype(np.float32)  # convert to float32 for operations
            mod_img = cv2.addWeighted(mod_img, contrast, np.zeros(mod_img.shape, mod_img.dtype), 0, brightness)
            mod_img = cv2.cvtColor(mod_img, cv2.COLOR_BGR2GRAY)
            mod_img = mod_img[..., np.newaxis]  # adds a channel dimension

            # Assign the modified image to the pre-allocated array
            image_data[index] = mod_img.astype(np.float16)
            labels.append(np.float16(i)) #number corresponding to image folder currently open
            index += 1  # Increment the index for the next image

            # apply and save horizontal flip
            mod_img_flip = cv2.flip(mod_img, 1)
            mod_img_flip = mod_img_flip[..., np.newaxis]
            image_data[index] = mod_img_flip.astype(np.float16)
            labels.append(np.float16(i))
            index += 1

            # apply some random rotation (2 images for original and 2 for flipped) and save
            pos_rot = random.randint(5, 30)
            neg_rot = random.randint(-30, -5)
            (h, w) = mod_img.shape[:2]
            (ctrX, ctrY) = (h // 2, w // 2)
            pos_rot_mat = cv2.getRotationMatrix2D((ctrX, ctrY), pos_rot, 1.0)
            neg_rot_mat = cv2.getRotationMatrix2D((ctrX, ctrY), neg_rot, 1.0)

            orig_pos_rot_img = cv2.warpAffine(mod_img, pos_rot_mat, (w, h))
            orig_neg_rot_img = cv2.warpAffine(mod_img, neg_rot_mat, (w, h))
            flip_pos_rot_img = cv2.warpAffine(mod_img_flip, pos_rot_mat, (w, h))
            flip_neg_rot_img = cv2.warpAffine(mod_img_flip, neg_rot_mat, (w, h))

            # Assign the rotated images to the pre-allocated array
            orig_pos_rot_img = orig_pos_rot_img[..., np.newaxis]
            image_data[index] = orig_pos_rot_img.astype(np.float16)
            labels.append(np.float16(i))
            index += 1

            orig_neg_rot_img = orig_neg_rot_img[..., np.newaxis]
            image_data[index] = orig_neg_rot_img.astype(np.float16)
            labels.append(np.float16(i))
            index += 1

            flip_pos_rot_img = flip_pos_rot_img[..., np.newaxis]
            image_data[index] = flip_pos_rot_img.astype(np.float16)
            labels.append(np.float16(i))
            index += 1

            flip_neg_rot_img = flip_neg_rot_img[..., np.newaxis]
            image_data[index] = flip_neg_rot_img.astype(np.float16)
            labels.append(np.float16(i))
            index += 1

            del temp_data, mod_img, mod_img_flip, orig_neg_rot_img, orig_pos_rot_img, flip_neg_rot_img, flip_pos_rot_img

    return image_data, labels