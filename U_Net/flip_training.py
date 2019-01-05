import PIL

import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import os
import matplotlib.image as mplimg

from tensorflow.contrib.slim.python.slim.data.tfexample_decoder import Image

TRAINING_PATH = 'Skin_Data_set/images/'
GROUNDTRUTH_PATH = 'Skin_Data_set/GroundTruth/'
TRAINING_TEST_PATH = 'Skin_Data_set/test/images_test/'
GROUNDTRUTH_TEST_PATH = 'Skin_Data_set/test/GroundTruth_test/'

TRAIN_SIZE = 2000
TEST_SIZE = 600


def flip_training():
    # load raw imgs
    data = load_images(TRAINING_PATH, TRAIN_SIZE)

    groundtruth = load_groundtruths(GROUNDTRUTH_PATH, TRAIN_SIZE)

    #data_test = load_images_test(TRAINING_TEST_PATH, TRAIN_SIZE)

    #groundtruth_test = load_groundtruths_test(GROUNDTRUTH_TEST_PATH, TRAIN_SIZE)

    if not os.path.exists('Skin_Data_set/images/'):
        os.makedirs('Skin_Data_set/images/')

    if not os.path.exists('Skin_Data_set/GroundTruth/'):
        os.makedirs('Skin_Data_set/GroundTruth/')

    if not os.path.exists('Skin_Data_set/test/images_test/'):
        os.makedirs('Skin_Data_set/test/images_test/')

    if not os.path.exists('Skin_Data_set/test/GroundTruth_test/'):
        os.makedirs('Skin_Data_set/test/GroundTruth_test/')



    for i in range(TRAIN_SIZE):

        print('Flipping train image' + str(i))
        data_img = data[i]
        groundtruth_img = groundtruth[i]#reshape(400, 400)
        data_name = "Skin_Data_set/flip_training/images/Image_%.3d" % (i*4 + 1) + '.png'
        groundtruth_name = "Skin_Data_set/flip_training/groundtruth/Image_%.3d" % (i*4 + 1) + '.png'
        misc.imsave(data_name, data_img)
        misc.imsave(groundtruth_name, groundtruth_img)
        for j in range(3):
            data_img = np.rot90(data_img)
            groundtruth_img = np.rot90(groundtruth_img)
            data_name = "Skin_Data_set/flip_training/images/Image_%.3d" % (i*4 + 2 + j) + '.png'
            groundtruth_name = "Skin_Data_set/flip_training/groundtruth/Image_%.3d" % (i*4 + 2+j) + '.png'
            misc.imsave(data_name, data_img)
            misc.imsave(groundtruth_name, groundtruth_img)

    """for i in range(TEST_SIZE):

        print('Flipping train image' + str(i))
        data_img_test = data_test[i]
        groundtruth_img_test = groundtruth_test[i]#reshape(400, 400)
        data_name_test = "Skin_Data_set/test/flip_training/images/Image_%.3d" % (i*4 + 1) + '.png'
        groundtruth_name_test = "Skin_Data_set/test/flip_training/groundtruth/Image_%.3d" % (i*4 + 1) + '.png'
        misc.imsave(data_name_test, data_img_test)
        misc.imsave(groundtruth_name_test, groundtruth_img_test)
        for j in range(3):
            data_img_test = np.rot90(data_img_test)
            groundtruth_img_test = np.rot90(groundtruth_img_test)
            data_name_test = "Skin_Data_set/test/flip_training/images/Image_%.3d" % (i*4 + 2 + j) + '.png'
            groundtruth_name_test = "Skin_Data_set/test/flip_training/groundtruth/Image_%.3d" % (i*4 + 2+j) + '.png'
            misc.imsave(data_name_test, data_img_test)
            misc.imsave(groundtruth_name_test, groundtruth_img_test)"""








def load_images(folder_path, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """
    
    """imgs = np.zeros(shape=[num_images, 400, 400, 3])
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img.reshape(400, 400, 3)
        else:
            print('File ' + image_path + ' does not exist')
    return imgs"""

    root_dir = "Skin_Data_set/"
    
    image_dir = root_dir + "images/"
    files = os.listdir(folder_path)
    n = len(files)
    print("Loading " + str(n) + " images")

    basewidth = 400
    baseheight = 400
    imgs = []
    for i in range(n):
        img = PIL.Image.open(folder_path + files[i])
        if img.size[0] > img.size[1]:
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize))
        if img.size[0] < img.size[1]:
            hpercent = (baseheight / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, baseheight))
        imgs.append(np.array(img))

    imgs = np.asarray(imgs)
    img = []
    for i in range(len(imgs)):
        if imgs[i].shape[0] < imgs[i].shape[1]:
            img.append(np.pad(imgs[i], ((0, 400 - imgs[i].shape[0]), (0, 0), (0, 0)), 'constant'))
            if img[i].shape[0] != 400 or img[i].shape[1] != 400:
                print(img[i].shape)
        elif imgs[i].shape[0] > imgs[i].shape[1]:
            img.append(np.pad(imgs[i], ((0, 0), (0, 400 - imgs[i].shape[1]), (0, 0)), 'constant'))
            if img[i].shape[0] != 400 or img[i].shape[1] != 400:
                print(img[i].shape)
        else:
            img.append(imgs[i])
    img = np.asarray(img)
    print ('shape img : ', img.shape)
    #imgs = imgs.reshape(num_images,400, 400, 3)
    #print ('if we doe a reshape :', imgs.shape)
    
    return img





def load_groundtruths(folder_path, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """
    
    """imgs = np.zeros(shape=[num_images, 400, 400, 1])
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img.reshape(400, 400, 1)
        else:
            print('File ' + image_path + ' does not exist')
    return imgs"""
    
    root_dir = "Skin_Data_set/"
    
    basewidth = 400
    baseheight = 400
    
    gt_dir = root_dir + "GroundTruth/"
    files = os.listdir(folder_path)
    n = len(files)
    print("Loading " + str(n) + " images")

    gts = []
    for i in range(n):
        gt = PIL.Image.open(folder_path + files[i])
        if gt.size[0] > gt.size[1]:
            wpercent = (basewidth / float(gt.size[0]))
            hsize = int((float(gt.size[1]) * float(wpercent)))
            gt = gt.resize((basewidth, hsize))
        if gt.size[0] < gt.size[1]:
            hpercent = (baseheight / float(gt.size[1]))
            wsize = int((float(gt.size[0]) * float(hpercent)))
            gt = gt.resize((wsize, baseheight))
        gts.append(np.array(gt))

    gts = np.asarray(gts)
    gt = []
    for i in range(len(gts)):
        if gts[i].shape[0] < gts[i].shape[1]:
            gt.append(np.pad(gts[i], ((0, 400 - gts[i].shape[0]), (0, 0)), 'constant'))
            if gt[i].shape[0] != 400 or gt[i].shape[1] != 400:
                print(gt[i].shape)
        elif gts[i].shape[0] > gts[i].shape[1]:
            gt.append(np.pad(gts[i], ((0, 0), (0, 400 - gts[i].shape[1])), 'constant'))
            if gt[i].shape[0] != 400 or gt[i].shape[1] != 400:
                print(gt[i].shape)
        else:
            gt.append(gts[i])
            
    gt = np.asarray(gt)
    
    print(gt.shape)
    return gt


def load_images_test(folder_path, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """

    """imgs = np.zeros(shape=[num_images, 400, 400, 3])
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img.reshape(400, 400, 3)
        else:
            print('File ' + image_path + ' does not exist')
    return imgs"""

    root_dir = "Skin_Data_set/"

    image_dir = root_dir + "images/"
    files = os.listdir(folder_path)
    n = len(files)
    print("Loading " + str(n) + " images")

    basewidth = 400
    baseheight = 400
    imgs = []
    for i in range(num_images):
        img = PIL.Image.open(folder_path + files[i])
        if img.size[0] > img.size[1]:
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize))
        if img.size[0] < img.size[1]:
            hpercent = (baseheight / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, baseheight))
        imgs.append(np.array(img))

    imgs = np.asarray(imgs)
    img = []
    for i in range(len(imgs)):
        if imgs[i].shape[0] < imgs[i].shape[1]:
            img.append(np.pad(imgs[i], ((0, 400 - imgs[i].shape[0]), (0, 0), (0, 0)), 'constant'))
            if img[i].shape[0] != 400 or img[i].shape[1] != 400:
                print(img[i].shape)
        elif imgs[i].shape[0] > imgs[i].shape[1]:
            img.append(np.pad(imgs[i], ((0, 0), (0, 400 - imgs[i].shape[1]), (0, 0)), 'constant'))
            if img[i].shape[0] != 400 or img[i].shape[1] != 400:
                print(img[i].shape)
        else:
            img.append(imgs[i])
    img = np.asarray(img)
    print('shape img : ', img.shape)
    # imgs = imgs.reshape(num_images,400, 400, 3)
    # print ('if we doe a reshape :', imgs.shape)

    return img


def load_groundtruths_test(folder_path, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """

    root_dir = "Skin_Data_set/"

    basewidth = 400
    baseheight = 400

    gt_dir = root_dir + "GroundTruth/"
    files = os.listdir(folder_path)
    n = len(files)
    print("Loading " + str(n) + " images")

    gts = []
    for i in range(n):
        gt = PIL.Image.open(folder_path + files[i])
        if gt.size[0] > gt.size[1]:
            wpercent = (basewidth / float(gt.size[0]))
            hsize = int((float(gt.size[1]) * float(wpercent)))
            gt = gt.resize((basewidth, hsize))
        if gt.size[0] < gt.size[1]:
            hpercent = (baseheight / float(gt.size[1]))
            wsize = int((float(gt.size[0]) * float(hpercent)))
            gt = gt.resize((wsize, baseheight))
        gts.append(np.array(gt))

    gts = np.asarray(gts)
    gt = []
    for i in range(len(gts)):
        if gts[i].shape[0] < gts[i].shape[1]:
            gt.append(np.pad(gts[i], ((0, 400 - gts[i].shape[0]), (0, 0)), 'constant'))
            if gt[i].shape[0] != 400 or gt[i].shape[1] != 400:
                print(gt[i].shape)
        elif gts[i].shape[0] > gts[i].shape[1]:
            gt.append(np.pad(gts[i], ((0, 0), (0, 400 - gts[i].shape[1])), 'constant'))
            if gt[i].shape[0] != 400 or gt[i].shape[1] != 400:
                print(gt[i].shape)
        else:
            gt.append(gts[i])

    gt = np.asarray(gt)

    print(gt.shape)
    return gt


if __name__ == '__main__':
    flip_training()
