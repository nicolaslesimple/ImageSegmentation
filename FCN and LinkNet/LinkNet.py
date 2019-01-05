 # convert RGB images x to grayscale using the formula for Y_linear in https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
def grayscale(x):
    x = x.astype('float32')/255
    x = np.piecewise(x, [x <= 0.04045, x > 0.04045],
                        [lambda x: x/12.92, lambda x: ((x + .055)/1.055)**2.4])
    return .2126 * x[:,:,0] + .7152 * x[:,:,1]  + .07152 * x[:,:,2]

def Import_images():
    """import all necessary images"""
    # Load the training set
    root_dir = "Data_set/"

    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = len(files)
    print("Loading " + str(n) + " images")
    imgs = np.asarray([mplimg.imread(image_dir + files[i]) for i in range(n)])

    gt_dir = root_dir + "GroundTruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = np.asarray([mplimg.imread(gt_dir + files[i]) for i in range(n)])

    # Conversion to RGB format
    img = imgs.copy()
    for i in range(len(imgs)):
        if img[i].shape[2] == 4:
            img[i] = (np.delete(img[i], 3, 2))

    # Conversion to grayscale image
    gt = gt_imgs.copy()
    gt_gray = []
    for i in range(len(gt_imgs)):
        if len(gt[i].shape) == 3:
            if gt[i].shape[2] == 4:
                gt[i] = (np.delete(gt[i], 3, 2))

        gt_gray.append(grayscale(gt[i]))
    gt_gray = np.asarray(gt_gray)

    for i in range(len(gt_gray)):
        for j in range(gt_gray[i].shape[0]):
            for k in range(gt_gray[i].shape[1]):
                if gt_gray[i][j][k] > 0.01:
                    gt_gray[i][j][k] = 1
                else:
                    gt_gray[i][j][k] = 0

                    # print (gt[i].shape)
                    # Save image in folder

                    #    for i in range (len(gt_gray)):
                    #       plt.imshow(gt_gray[i])
                    #      plt.show()
                    #     plt.imshow(img[i])
    # plt.show()

    return img, gt_gray

###################################################### Imports #########################################################

import numpy as np
import random
import os.path
import time
import scipy.misc
import re
import tensorflow as tf
import sys
import cv2
from patches import extract_patches_from_dir
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#import matplotlib.pyplot as plt
########################################################################################################################
################################################ Global parameters #####################################################
print('1 done')
network_name = 'network'
number_of_epoches = 1000
learning_rate = 0.0003
batch_size = 37
#batch_size = 5
image_height = 384
image_width = 512
########################################################################################################################
################################################ Data acquisition ######################################################
print('2 done')
#   Train set (Original RGB image, Greyscale image)
train_patches = extract_patches_from_dir('Data_set/images',[16,16])
nul_1, nul_2 = os.walk(os.path.join('Data_set')).__next__()
#nul_1, nul_2, train_patches = os.walk(os.path.join('datasets', network_name, 'philippe-plus-ISIC_train_patches')).__next__()
train_patches_count = len(train_patches)
print('train_patches_count : ',train_patches_count)
#train_patches_count = 20
number_patches_total = np.multiply(int(np.true_divide(train_patches_count,batch_size)),batch_size)
########################################################################################################################
##################################################### Network ##########################################################
print('3 done')

with tf.device('/gpu:0'):

    print('start initing')
    # x (Input: original RGB image), x_labels (Greyscale image)
    x = tf.placeholder(tf.float32, shape=(None, image_height, image_width, 3), name="original_image")
    x_labels = tf.placeholder(tf.float32, shape=(None, image_height, image_width, 1), name="original_image_segmentation_label")

    ###########################################################
    ######################### Network #########################
    # c_1 convolution
    c_1_weights = tf.get_variable("c_1_weights", shape=[7, 7, 3, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_1_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_1_filter = tf.nn.conv2d(x, c_1_weights, [1, 2, 2, 1], padding='SAME')
    c_1 = tf.nn.relu(c_1_filter + c_1_biases)

    # p_1 pooling
    p_1 = tf.nn.max_pool(c_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    ###########################################################
    ######################### Encoder #########################
    ####################### Encoder Block 1
    # c_2 convolution
    c_2_weights = tf.get_variable("c_2_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_2_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_2_filter = tf.nn.conv2d(p_1, c_2_weights, [1, 2, 2, 1], padding='SAME')
    c_2 = tf.nn.relu(c_2_filter + c_2_biases)

    # c_3 convolution
    c_3_weights = tf.get_variable("c_3_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_3_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_3_filter = tf.nn.conv2d(c_2, c_3_weights, [1, 1, 1, 1], padding='SAME')
    c_3 = tf.nn.relu(c_3_filter + c_3_biases)

    c_transition_encoder_1_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name='c_transition_encoder_1_weights')
    encoder_block_1_temp = c_3 + tf.nn.relu(tf.nn.conv2d(p_1, filter=c_transition_encoder_1_weights, strides=[1, 2, 2, 1], padding="SAME"))

    # c_4 convolution
    c_4_weights = tf.get_variable("c_4_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_4_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_4_filter = tf.nn.conv2d(encoder_block_1_temp, c_4_weights, [1, 1, 1, 1], padding='SAME')
    c_4 = tf.nn.relu(c_4_filter + c_4_biases)

    # c_5 convolution
    c_5_weights = tf.get_variable("c_5_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_5_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_5_filter = tf.nn.conv2d(c_4, c_5_weights, [1, 1, 1, 1], padding='SAME')
    c_5 = tf.nn.relu(c_5_filter + c_5_biases)

    encoder_block_1 = c_5 + encoder_block_1_temp
    ####################### Encoder Block 2
    # c_6 convolution
    c_6_weights = tf.get_variable("c_6_weights", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_6_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_6_filter = tf.nn.conv2d(encoder_block_1, c_6_weights, [1, 2, 2, 1], padding='SAME')
    c_6 = tf.nn.relu(c_6_filter + c_6_biases)

    # c_7 convolution
    c_7_weights = tf.get_variable("c_7_weights", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_7_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_7_filter = tf.nn.conv2d(c_6, c_7_weights, [1, 1, 1, 1], padding='SAME')
    c_7 = tf.nn.relu(c_7_filter + c_7_biases)

    c_transition_encoder_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='c_transition_encoder_2_weights')
    encoder_block_2_temp = c_7 + tf.nn.relu(tf.nn.conv2d(encoder_block_1, filter=c_transition_encoder_2_weights, strides=[1, 2, 2, 1], padding="SAME"))

    # c_8 convolution
    c_8_weights = tf.get_variable("c_8_weights", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_8_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_8_filter = tf.nn.conv2d(encoder_block_2_temp, c_8_weights, [1, 1, 1, 1], padding='SAME')
    c_8 = tf.nn.relu(c_8_filter + c_8_biases)

    # c_9 convolution
    c_9_weights = tf.get_variable("c_9_weights", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_9_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_9_filter = tf.nn.conv2d(c_8, c_9_weights, [1, 1, 1, 1], padding='SAME')
    c_9 = tf.nn.relu(c_9_filter + c_9_biases)

    encoder_block_2 = c_9 + encoder_block_2_temp
    ####################### Encoder Block 3
    # c_10 convolution
    c_10_weights = tf.get_variable("c_10_weights", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    c_10_biases = tf.Variable(tf.zeros([256]), tf.float32)
    c_10_filter = tf.nn.conv2d(encoder_block_2, c_10_weights, [1, 2, 2, 1], padding='SAME')
    c_10 = tf.nn.relu(c_10_filter + c_10_biases)

    # c_11 convolution
    c_11_weights = tf.get_variable("c_11_weights", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    c_11_biases = tf.Variable(tf.zeros([256]), tf.float32)
    c_11_filter = tf.nn.conv2d(c_10, c_11_weights, [1, 1, 1, 1], padding='SAME')
    c_11 = tf.nn.relu(c_11_filter + c_11_biases)

    c_transition_encoder_3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name='c_transition_encoder_3_weights')
    encoder_block_3_temp = c_11 + tf.nn.relu(tf.nn.conv2d(encoder_block_2, filter=c_transition_encoder_3_weights, strides=[1, 2, 2, 1], padding="SAME"))

    # c_12 convolution
    c_12_weights = tf.get_variable("c_12_weights", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    c_12_biases = tf.Variable(tf.zeros([256]), tf.float32)
    c_12_filter = tf.nn.conv2d(encoder_block_3_temp, c_12_weights, [1, 1, 1, 1], padding='SAME')
    c_12 = tf.nn.relu(c_12_filter + c_12_biases)

    # c_13 convolution
    c_13_weights = tf.get_variable("c_13_weights", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    c_13_biases = tf.Variable(tf.zeros([256]), tf.float32)
    c_13_filter = tf.nn.conv2d(c_12, c_13_weights, [1, 1, 1, 1], padding='SAME')
    c_13 = tf.nn.relu(c_13_filter + c_13_biases)

    encoder_block_3 = c_13 + encoder_block_3_temp
    ####################### Encoder Block 4
    # c_14 convolution
    c_14_weights = tf.get_variable("c_14_weights", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    c_14_biases = tf.Variable(tf.zeros([512]), tf.float32)
    c_14_filter = tf.nn.conv2d(encoder_block_3, c_14_weights, [1, 2, 2, 1], padding='SAME')
    c_14 = tf.nn.relu(c_14_filter + c_14_biases)

    # c_15 convolution
    c_15_weights = tf.get_variable("c_15_weights", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    c_15_biases = tf.Variable(tf.zeros([512]), tf.float32)
    c_15_filter = tf.nn.conv2d(c_14, c_15_weights, [1, 1, 1, 1], padding='SAME')
    c_15 = tf.nn.relu(c_15_filter + c_15_biases)

    c_transition_encoder_4_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1), name='c_transition_encoder_4_weights')
    encoder_block_4_temp = c_15 + tf.nn.relu(tf.nn.conv2d(encoder_block_3, filter=c_transition_encoder_4_weights, strides=[1, 2, 2, 1], padding="SAME"))

    # c_16 convolution
    c_16_weights = tf.get_variable("c_16_weights", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    c_16_biases = tf.Variable(tf.zeros([512]), tf.float32)
    c_16_filter = tf.nn.conv2d(encoder_block_4_temp, c_16_weights, [1, 1, 1, 1], padding='SAME')
    c_16 = tf.nn.relu(c_16_filter + c_16_biases)

    # c_17 convolution
    c_17_weights = tf.get_variable("c_17_weights", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    c_17_biases = tf.Variable(tf.zeros([512]), tf.float32)
    c_17_filter = tf.nn.conv2d(c_16, c_17_weights, [1, 1, 1, 1], padding='SAME')
    c_17 = tf.nn.relu(c_17_filter + c_17_biases)

    encoder_block_4 = c_17 + encoder_block_4_temp
    ###########################################################
    ######################### Decoder #########################
    ####################### Decoder Block 4
    # c_18 convolution
    c_18_weights = tf.get_variable("c_18_weights", shape=[1, 1, 512, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_18_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_18_filter = tf.nn.conv2d(encoder_block_4, c_18_weights, [1, 1, 1, 1], padding='SAME')
    c_18 = tf.nn.relu(c_18_filter + c_18_biases)

    # c_19 convolution
    c_19_weights = tf.get_variable("c_19_weights", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_19_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_19_filter = tf.nn.conv2d_transpose(c_18, c_19_weights, [batch_size, c_18.get_shape().as_list()[1], c_18.get_shape().as_list()[2], 128], [1, 1, 1, 1], padding='SAME', name=None)
    c_19 = tf.nn.relu(c_19_filter + c_19_biases)

    u_1 = tf.image.resize_bilinear(c_19, (c_19.get_shape().as_list()[1] * 2, c_19.get_shape().as_list()[2] * 2))

    # c_20 convolution
    c_20_weights = tf.get_variable("c_20_weights", shape=[1, 1, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    c_20_biases = tf.Variable(tf.zeros([256]), tf.float32)
    c_20_filter = tf.nn.conv2d(u_1, c_20_weights, [1, 1, 1, 1], padding='SAME')
    c_20 = tf.nn.relu(c_20_filter + c_20_biases) + encoder_block_3
    ####################### Decoder Block 3
    # c_21 convolution
    c_21_weights = tf.get_variable("c_21_weights", shape=[1, 1, 256, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_21_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_21_filter = tf.nn.conv2d(c_20, c_21_weights, [1, 1, 1, 1], padding='SAME')
    c_21 = tf.nn.relu(c_21_filter + c_21_biases)

    # c_22 convolution
    c_22_weights = tf.get_variable("c_22_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_22_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_22_filter = tf.nn.conv2d_transpose(c_21, c_22_weights, [batch_size, c_21.get_shape().as_list()[1], c_21.get_shape().as_list()[2], 64], [1, 1, 1, 1], padding='SAME', name=None)
    c_22 = tf.nn.relu(c_22_filter + c_22_biases)

    u_2 = tf.image.resize_bilinear(c_22, (c_22.get_shape().as_list()[1] * 2, c_22.get_shape().as_list()[2] * 2))

    # c_23 convolution
    c_23_weights = tf.get_variable("c_23_weights", shape=[1, 1, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    c_23_biases = tf.Variable(tf.zeros([128]), tf.float32)
    c_23_filter = tf.nn.conv2d(u_2, c_23_weights, [1, 1, 1, 1], padding='SAME')
    c_23 = tf.nn.relu(c_23_filter + c_23_biases) + encoder_block_2
    ####################### Decoder Block 2
    # c_24 convolution
    c_24_weights = tf.get_variable("c_24_weights", shape=[1, 1, 128, 32], initializer=tf.contrib.layers.xavier_initializer())
    c_24_biases = tf.Variable(tf.zeros([32]), tf.float32)
    c_24_filter = tf.nn.conv2d(c_23, c_24_weights, [1, 1, 1, 1], padding='SAME')
    c_24 = tf.nn.relu(c_24_filter + c_24_biases)

    # c_25 convolution
    c_25_weights = tf.get_variable("c_25_weights", shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
    c_25_biases = tf.Variable(tf.zeros([32]), tf.float32)
    c_25_filter = tf.nn.conv2d_transpose(c_24, c_25_weights, [batch_size, c_24.get_shape().as_list()[1], c_24.get_shape().as_list()[2], 32], [1, 1, 1, 1], padding='SAME', name=None)
    c_25 = tf.nn.relu(c_25_filter + c_25_biases)

    u_3 = tf.image.resize_bilinear(c_25, (c_25.get_shape().as_list()[1] * 2, c_25.get_shape().as_list()[2] * 2))

    # c_26 convolution
    c_26_weights = tf.get_variable("c_26_weights", shape=[1, 1, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_26_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_26_filter = tf.nn.conv2d(u_3, c_26_weights, [1, 1, 1, 1], padding='SAME')
    c_26 = tf.nn.relu(c_26_filter + c_26_biases) + encoder_block_1
    ####################### Decoder Block 1
    # c_27 convolution
    c_27_weights = tf.get_variable("c_27_weights", shape=[1, 1, 64, 16], initializer=tf.contrib.layers.xavier_initializer())
    c_27_biases = tf.Variable(tf.zeros([16]), tf.float32)
    c_27_filter = tf.nn.conv2d(c_26, c_27_weights, [1, 1, 1, 1], padding='SAME')
    c_27 = tf.nn.relu(c_27_filter + c_27_biases)

    # c_28 convolution
    c_28_weights = tf.get_variable("c_28_weights", shape=[3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
    c_28_biases = tf.Variable(tf.zeros([16]), tf.float32)
    c_28_filter = tf.nn.conv2d_transpose(c_27, c_28_weights, [batch_size, c_27.get_shape().as_list()[1], c_27.get_shape().as_list()[2], 16], [1, 1, 1, 1], padding='SAME', name=None)
    c_28 = tf.nn.relu(c_28_filter + c_28_biases)

    u_4 = tf.image.resize_bilinear(c_28, (c_28.get_shape().as_list()[1] * 2, c_28.get_shape().as_list()[2] * 2))

    # c_29 convolution
    c_29_weights = tf.get_variable("c_29_weights", shape=[1, 1, 16, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_29_biases = tf.Variable(tf.zeros([64]), tf.float32)
    c_29_filter = tf.nn.conv2d(u_4, c_29_weights, [1, 1, 1, 1], padding='SAME')
    c_29 = tf.nn.relu(c_29_filter + c_29_biases)
    ###########################################################
    ###########################################################
    # c_30 convolution
    c_30_weights = tf.get_variable("c_30_weights", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_30_biases = tf.Variable(tf.zeros([32]), tf.float32)
    c_30_filter = tf.nn.conv2d_transpose(c_29, c_30_weights, [batch_size, c_29.get_shape().as_list()[1], c_29.get_shape().as_list()[2], 32], [1, 1, 1, 1], padding='SAME', name=None)
    c_30 = tf.nn.relu(c_30_filter + c_30_biases)

    u_5 = tf.image.resize_bilinear(c_30, (c_30.get_shape().as_list()[1] * 2, c_30.get_shape().as_list()[2] * 2))

    # c_31 convolution
    c_31_weights = tf.get_variable("c_31_weights", shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
    c_31_biases = tf.Variable(tf.zeros([32]), tf.float32)
    c_31_filter = tf.nn.conv2d(u_5, c_31_weights, [1, 1, 1, 1], padding='SAME')
    c_31 = tf.nn.relu(c_31_filter + c_31_biases)

    # c_32 convolution
    c_32_weights = tf.get_variable("c_32_weights", shape=[2, 2, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    c_32_biases = tf.Variable(tf.zeros([1]), tf.float32)
    c_32_filter = tf.nn.conv2d_transpose(c_31, c_32_weights, [batch_size, c_31.get_shape().as_list()[1], c_31.get_shape().as_list()[2], 1], [1, 1, 1, 1], padding='SAME', name=None)
    c_32 = tf.nn.sigmoid(c_32_filter + c_32_biases)

    u_6 = tf.image.resize_bilinear(c_32, (c_32.get_shape().as_list()[1] * 2, c_32.get_shape().as_list()[2] * 2))
    ###########################################################
    ########################### Loss function
    X_L = tf.squeeze(x_labels)
    D_5 = tf.squeeze(u_6)
    Num = tf.reduce_sum(X_L * D_5)
    De_num_1 = tf.reduce_sum(X_L * X_L)
    De_num_2 = tf.reduce_sum(D_5 * D_5)
    loss = 1 - (Num/(De_num_1+De_num_2-Num))
    ###########################################################
    ########################### Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    print('end initing')
########################################################################################################################
##################################################### Session ##########################################################
print('4 done')
path_results = os.path.join('results', 'experiment_'+time.strftime("%d_%m_%Y")+'_'+time.strftime("%H_%M_%S")+'_')
if '-save' in sys.argv:
    if not os.path.exists(path_results) and path_results:
        os.makedirs(path_results)
        print('Folder {} has been created'.format(path_results))

iteration_number = int(number_patches_total/batch_size)
init_op = tf.initialize_all_variables()
print('5 done')

train_patches_path = os.path.join('datasets', network_name, 'philippe-plus-ISIC_2016_train_patches')
train_patches_label_path = os.path.join('datasets', network_name, 'philippe-plus-ISIC_2016_train_patches_label')
Network_train_patches = list(np.zeros((batch_size, image_height, image_width, 3)).astype(np.float32))
Network_train_label_patches = list(np.zeros((batch_size, image_height, image_width)).astype(np.float32))

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init_op)
    #sess.graph.finalize()
    print("Initialized")
    if '-save' in sys.argv:
        np.savetxt(os.path.join(path_results, 'Loss_avg_iter_train.txt'), [], delimiter=" ", fmt="%s")

    for epoch in range(number_of_epoches):
        Loss_avg_iter_train = []
        order = list(random.sample(range(0, number_patches_total), number_patches_total))
        step = 0
        for count in range(iteration_number):
            for load_ind in range(batch_size):
                Network_train_patches[load_ind] = cv2.resize(
                    cv2.cvtColor(cv2.imread(train_patches_path + '/' + train_patches[order[step]]), cv2.COLOR_BGR2RGB),
                    (image_width, image_height))

                Network_train_label_patches[load_ind] = cv2.resize(
                    cv2.imread(train_patches_label_path + '/' + train_patches[order[step]], 0),
                    (image_width, image_height))
                step = step + 1

            train_set_ref = np.array(Network_train_patches)/ 255.0
            #train_set_ref = train_set_ref.reshape(batch_size, image_height, image_width, 3)

            train_set_ref_label = np.array(Network_train_label_patches)/ 255.0
            #train_set_ref_label = train_set_ref_label.reshape(batch_size, image_height, image_width, 1)
            train_set_ref_label = np.expand_dims(train_set_ref_label, axis=-1)

            '''fig = plt.figure()
            image_original = plt.subplot("121")
            image_original.set_title("Original Image")
            image_original.imshow(np.squeeze(train_set_ref))
            image_original_label = plt.subplot("122")
            image_original_label.set_title("Original Image Label")
            image_original_label.imshow(np.squeeze(train_set_ref_label))
            plt.show()'''

            if count < (iteration_number-1):
                _, l = sess.run([optimizer, loss], {x: train_set_ref, x_labels: train_set_ref_label})
                print("epoch ", epoch, "      iteration ", count, "      loss ", l)
                Loss_avg_iter_train.append(l)

        _, l, c_1_W, c_2_W, c_3_W, c_4_W, c_5_W, c_6_W, c_7_W, c_8_W, c_9_W, c_10_W, c_11_W, c_12_W, c_13_W, c_14_W,\
        c_15_W, c_16_W, c_17_W, c_18_W, c_19_W, c_20_W, c_21_W, c_22_W, c_23_W, c_24_W, c_25_W, c_26_W, c_27_W,\
        c_28_W, c_29_W, c_30_W, c_31_W, c_32_W, c_1_B, c_2_B, c_3_B, c_4_B, c_5_B, c_6_B, c_7_B, c_8_B, c_9_B,\
        c_10_B, c_11_B, c_12_B, c_13_B, c_14_B, c_15_B, c_16_B, c_17_B, c_18_B, c_19_B, c_20_B, c_21_B, c_22_B,\
        c_23_B, c_24_B, c_25_B, c_26_B, c_27_B, c_28_B, c_29_B, c_30_B, c_31_B, c_32_B, c_transition_encoder_1_W,\
        c_transition_encoder_2_W, c_transition_encoder_3_W, c_transition_encoder_4_W\
            = sess.run([optimizer, loss, c_1_weights, c_2_weights, c_3_weights, c_4_weights, c_5_weights,
                        c_6_weights, c_7_weights, c_8_weights, c_9_weights, c_10_weights, c_11_weights,
                        c_12_weights, c_13_weights, c_14_weights, c_15_weights, c_16_weights, c_17_weights,
                        c_18_weights, c_19_weights, c_20_weights, c_21_weights, c_22_weights, c_23_weights,
                        c_24_weights, c_25_weights, c_26_weights, c_27_weights, c_28_weights, c_29_weights,
                        c_30_weights, c_31_weights, c_32_weights, c_1_biases, c_2_biases, c_3_biases, c_4_biases,
                        c_5_biases, c_6_biases, c_7_biases, c_8_biases, c_9_biases, c_10_biases, c_11_biases,
                        c_12_biases, c_13_biases, c_14_biases, c_15_biases, c_16_biases, c_17_biases, c_18_biases,
                        c_19_biases, c_20_biases, c_21_biases, c_22_biases, c_23_biases, c_24_biases, c_25_biases,
                        c_26_biases, c_27_biases, c_28_biases, c_29_biases, c_30_biases, c_31_biases,
                        c_32_biases, c_transition_encoder_1_weights, c_transition_encoder_2_weights,
                        c_transition_encoder_3_weights, c_transition_encoder_4_weights], {x: train_set_ref, x_labels: train_set_ref_label})
        print("final :  ","epoch ", epoch, "      iteration ", count, "      loss ", l)
        Loss_avg_iter_train.append(l)
        if '-save' in sys.argv:
            with open(os.path.join(path_results, 'Loss_avg_iter_train.txt'), 'a') as file:
                file.writelines('\n%s' %(Loss_avg_iter_train))

        if '-save' in sys.argv:
            if epoch%1 == 0:
                print('saving !')
                np.save(os.path.join(path_results, "c_1_weights"), c_1_W)
                np.save(os.path.join(path_results, "c_2_weights"), c_2_W)
                np.save(os.path.join(path_results, "c_3_weights"), c_3_W)
                np.save(os.path.join(path_results, "c_4_weights"), c_4_W)
                np.save(os.path.join(path_results, "c_5_weights"), c_5_W)

                np.save(os.path.join(path_results, "c_6_weights"), c_6_W)
                np.save(os.path.join(path_results, "c_7_weights"), c_7_W)
                np.save(os.path.join(path_results, "c_8_weights"), c_8_W)
                np.save(os.path.join(path_results, "c_9_weights"), c_9_W)
                np.save(os.path.join(path_results, "c_10_weights"), c_10_W)

                np.save(os.path.join(path_results, "c_11_weights"), c_11_W)
                np.save(os.path.join(path_results, "c_12_weights"), c_12_W)
                np.save(os.path.join(path_results, "c_13_weights"), c_13_W)
                np.save(os.path.join(path_results, "c_14_weights"), c_14_W)
                np.save(os.path.join(path_results, "c_15_weights"), c_15_W)

                np.save(os.path.join(path_results, "c_16_weights"), c_16_W)
                np.save(os.path.join(path_results, "c_17_weights"), c_17_W)
                np.save(os.path.join(path_results, "c_18_weights"), c_18_W)
                np.save(os.path.join(path_results, "c_19_weights"), c_19_W)
                np.save(os.path.join(path_results, "c_20_weights"), c_20_W)

                np.save(os.path.join(path_results, "c_21_weights"), c_21_W)
                np.save(os.path.join(path_results, "c_22_weights"), c_22_W)
                np.save(os.path.join(path_results, "c_23_weights"), c_23_W)
                np.save(os.path.join(path_results, "c_24_weights"), c_24_W)
                np.save(os.path.join(path_results, "c_25_weights"), c_25_W)

                np.save(os.path.join(path_results, "c_26_weights"), c_26_W)
                np.save(os.path.join(path_results, "c_27_weights"), c_27_W)
                np.save(os.path.join(path_results, "c_28_weights"), c_28_W)
                np.save(os.path.join(path_results, "c_29_weights"), c_29_W)
                np.save(os.path.join(path_results, "c_30_weights"), c_30_W)

                np.save(os.path.join(path_results, "c_31_weights"), c_31_W)
                np.save(os.path.join(path_results, "c_32_weights"), c_32_W)

                np.save(os.path.join(path_results, "c_1_biases"), c_1_B)
                np.save(os.path.join(path_results, "c_2_biases"), c_2_B)
                np.save(os.path.join(path_results, "c_3_biases"), c_3_B)
                np.save(os.path.join(path_results, "c_4_biases"), c_4_B)
                np.save(os.path.join(path_results, "c_5_biases"), c_5_B)

                np.save(os.path.join(path_results, "c_6_biases"), c_6_B)
                np.save(os.path.join(path_results, "c_7_biases"), c_7_B)
                np.save(os.path.join(path_results, "c_8_biases"), c_8_B)
                np.save(os.path.join(path_results, "c_9_biases"), c_9_B)
                np.save(os.path.join(path_results, "c_10_biases"), c_10_B)

                np.save(os.path.join(path_results, "c_11_biases"), c_11_B)
                np.save(os.path.join(path_results, "c_12_biases"), c_12_B)
                np.save(os.path.join(path_results, "c_13_biases"), c_13_B)
                np.save(os.path.join(path_results, "c_14_biases"), c_14_B)
                np.save(os.path.join(path_results, "c_15_biases"), c_15_B)

                np.save(os.path.join(path_results, "c_16_biases"), c_16_B)
                np.save(os.path.join(path_results, "c_17_biases"), c_17_B)
                np.save(os.path.join(path_results, "c_18_biases"), c_18_B)
                np.save(os.path.join(path_results, "c_19_biases"), c_19_B)
                np.save(os.path.join(path_results, "c_20_biases"), c_20_B)

                np.save(os.path.join(path_results, "c_21_biases"), c_21_B)
                np.save(os.path.join(path_results, "c_22_biases"), c_22_B)
                np.save(os.path.join(path_results, "c_23_biases"), c_23_B)
                np.save(os.path.join(path_results, "c_24_biases"), c_24_B)
                np.save(os.path.join(path_results, "c_25_biases"), c_25_B)

                np.save(os.path.join(path_results, "c_26_biases"), c_26_B)
                np.save(os.path.join(path_results, "c_27_biases"), c_27_B)
                np.save(os.path.join(path_results, "c_28_biases"), c_28_B)
                np.save(os.path.join(path_results, "c_29_biases"), c_29_B)
                np.save(os.path.join(path_results, "c_30_biases"), c_30_B)

                np.save(os.path.join(path_results, "c_31_biases"), c_31_B)
                np.save(os.path.join(path_results, "c_32_biases"), c_32_B)

                np.save(os.path.join(path_results, "c_transition_encoder_1_weights"), c_transition_encoder_1_W)
                np.save(os.path.join(path_results, "c_transition_encoder_2_weights"), c_transition_encoder_2_W)

                np.save(os.path.join(path_results, "c_transition_encoder_3_weights"), c_transition_encoder_3_W)
                np.save(os.path.join(path_results, "c_transition_encoder_4_weights"), c_transition_encoder_4_W)
print('hello')
############################################################################################










