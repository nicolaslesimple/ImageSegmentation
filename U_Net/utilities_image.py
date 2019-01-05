import os
import matplotlib.image as mpimg
from numpy.linalg import linalg
from scipy import misc
import tensorflow as tf


import numpy as np



PIXEL_DEPTH = 255

def PCA_augmentation (img):
    print('PCA color : Red augmentation begin ...')
    res = np.zeros(shape=(1, 3))
    for i in range(img.shape[0]):
        # re-shape to make list of RGB vectors.
        arr = img[i].reshape((400 * 400), 3)
        # consolidate RGB vectors of all images
        res = np.concatenate((res, arr), axis=0)
    res = np.delete(res, (0), axis=0)
    mean = res.mean(axis=0)
    res = res - mean  # allow the pca to work well
    R = np.cov(res, rowvar=False)  # covariance matrix
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the best 3 eigenvectors (3 is desired dimension
    # of rescaled data array)
    evecs = evecs[:, :3]
    # make a matrix with the three eigenvectors as its columns.
    evecs_mat = np.column_stack((evecs))
    # PCA
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    m = np.dot(evecs.T, res.T).T
    # we need to add multiples of the found principal components with magnitudes proportional to
    # the corresponding eigenvalues times a random variable drawn from a Gaussian distribution
    # with mean zero and standard deviation 0.1.
    for i in range(img.shape[0]):
        img[i] = img[i] / 255.0
        mu = 0
        sigma = 0.1
        feature_vec = np.matrix(evecs_mat)

        # 3 x 1 scaled eigenvalue matrix
        se = np.zeros((3, 1))
        se[0][0] = np.random.normal(mu, sigma) * evals[0]
        se[1][0] = np.random.normal(mu, sigma) * evals[1]
        se[2][0] = np.random.normal(mu, sigma) * evals[2]
        se = np.matrix(se)
        val = feature_vec * se

        # Parse through every pixel value.
        for l in range(img[i].shape[0]):
            for j in range(img[i].shape[1]):
                # Parse through every dimension.
                for k in range(img[i].shape[2]):
                    img[i][l, j, k] = float(img[i][l, j, k]) + float(val[k])

    print('PCA color : Red augmentation has been done')
    return img

# Last image index is excluded
def load_images(folder_path, num_images,red_augmentation=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """
    imgs = np.zeros(shape=[num_images//4, 400, 400, 3])
    for i in range(1, (num_images + 1)//4):
        image_name = "Image_%.3d" % ((i*4+1))
        image_path = folder_path +image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img#.reshape(400, 400, 3)
        else:
            print('File ' + image_path + ' does not exist')

    if red_augmentation == True:
        imgs = PCA_augmentation(imgs)

    return imgs

# Last image index is excluded
def load_images_cerv(folder_path, num_images,red_augmentation=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """
    imgs = np.zeros(shape=[num_images, 400, 400, 3])
    for i in range(1, num_images):
        image_name = "Image_%.3d" % (i)
        image_path = folder_path +image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img#.reshape(400, 400, 3)
        else:
            print('File ' + image_path + ' does not exist')

    if red_augmentation == True:
        imgs = PCA_augmentation(imgs)

    return imgs



def load_test_images(test_path, num_images):
    imgs = np.zeros(shape=[num_images, 400, 400, 3])
    for i in range(0, int(num_images/4) + 1):
        for j in range(1, 5):
            image_path = test_path + 'test_' + str(i) + '_' + str(j) + '.png'
            if os.path.isfile(image_path):
                print('Loading ' + image_path)
                img = mpimg.imread(image_path)
                imgs[(i-1)*4 + (j-1)] = img.reshape(400, 400, 3)
            else:
                print('File ' + image_path + ' does not exist')
    return imgs

def grayscale(x):
    x = x.astype('float32')/255
    x = np.piecewise(x, [x <= 0.04045, x > 0.04045],
                        [lambda x: x/12.92, lambda x: ((x + .055)/1.055)**2.4])
    return .2126 * x[:,:,0] + .7152 * x[:,:,1]  + .07152 * x[:,:,2]


def convert_image_to_hot(img):
    hot_img = np.zeros([400, 400, 2], dtype = float)
    if len(img.shape)==3:
        img=grayscale(img)
    for i in range(hot_img.shape[0]):
        for j in range(hot_img.shape[1]):
            if img[i,j] < 0.5:
                hot_img[i,j,0] = 1.0
                hot_img[i,j,1] = 0.0
            else:
                hot_img[i,j,0] = 0.0
                hot_img[i,j,1] = 1.0

    return hot_img


def load_groundtruths(folder_path, num_images):
    """Extract the groundtruth images into a 4D tensor [image index, y, x, channels].
        Indices are from 0."""
    imgs = np.zeros(shape=[num_images//4, 400, 400, 2])
    j=0
    for i in range(0, (num_images //4)+1):
        j=j+1
        image_name = "Image_%.3d" % ((i*4+1))
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)
            # See if it is better to use dtype = int
            hot_img = convert_image_to_hot(img)
            imgs[i-1]=(hot_img)
        else:
            print('File ' + image_path + ' does not exist')
    #imgs = np.around(imgs) # Uncomment if we want to round values.
    #imgs_array = np.asarray(imgs)
    return imgs#_array

def load_groundtruths_cerv(folder_path, num_images):
    """Extract the groundtruth images into a 4D tensor [image index, y, x, channels].
        Indices are from 0."""
    imgs = np.zeros(shape=[num_images, 400, 400, 2])
    j=0
    for i in range(0, num_images):
        j=j+1
        image_name = "Image_%.3d" % (i)
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)
            # See if it is better to use dtype = int
            hot_img = convert_image_to_hot(img)
            imgs[i-1]=(hot_img)
        else:
            print('File ' + image_path + ' does not exist')
    #imgs = np.around(imgs) # Uncomment if we want to round values.
    #imgs_array = np.asarray(imgs)
    return imgs#_array



def compare_proba_pred(prediction, original_img, save_path):
    #img_prediction = np.argmax(prediction, axis = 3)
    img_prediction = prediction[:,:,:,1]
    img_prediction = np.squeeze(img_prediction, axis = 0)

    min_value = np.amin(img_prediction)
    max_value = np.amax(img_prediction)
    img_prediction = (img_prediction - min_value) * (255.0 / (max_value - min_value))

    #print('PTEDICTIOM',img_prediction.shape)
    #print(img_prediction)

    if len(original_img.shape)==4:
        concatenated = concatenate_images(original_img[0], img_prediction)
    elif len(original_img.shape)==3:
        concatenated = concatenate_images(original_img, img_prediction)

    save_image(concatenated, save_path)



def merge_400_400_pred(array_pred):

    crop = 5

    merged_img = np.zeros((608, 608))

    #crop1 = list(range(0, 400 - crop))
    crop1 = np.arange(0, 400 -crop)
    crop2 = list(range(208+crop, 608))
    # Summing all contributions
    merged_img[:400-crop, :400-crop] += array_pred[0][:(400-crop), :(400-crop), 1]
    merged_img[208+crop:608, 0:400-crop] += array_pred[1][crop:, :(400-crop), 1]
    merged_img[0:400-crop, 208+crop:608] += array_pred[2][:(400-crop), crop:,1]
    merged_img[208+crop:608, 208+crop:608] += array_pred[3][crop:, crop:, 1]


    crop_a = list(range(0, 208+crop))
    crop_b = list(range(400-crop, 608))
    crop_c = list(range(208+crop, 400-crop))


    # Averaging
    merged_img[:208+crop, 208+crop:400-crop] = merged_img[:208+crop, 208+crop:400-crop] / 2
    merged_img[400-crop:608, 208+crop:400-crop] = merged_img[400-crop:608, 208+crop:400-crop] / 2
    merged_img[208+crop:400-crop, :208+crop] = merged_img[208+crop:400-crop, :208+crop] / 2
    merged_img[208+crop:400-crop, 400-crop:608] = merged_img[208+crop:400-crop, 400-crop:608] / 2
    merged_img[208+crop:400-crop, 208+crop:400-crop] = merged_img[208+crop:400-crop, 208+crop:400-crop] / 4

    return merged_img



def save_image(img, save_path):
    misc.imsave(save_path, img)
    print('Saved image in', save_path)






def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
       # print (img8.shape)
       # print (gt_img_3c.shape)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx = 0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value*PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


