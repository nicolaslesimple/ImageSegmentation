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
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#import matplotlib.pyplot as plt
########################################################################################################################
################################################ Global parameters #####################################################
print('1 done')
network_name = 'network'
number_of_epoches = 1000
learning_rate = 0.0003
batch_size = 37
#batch_size = 5
image_height = 400
image_width = 400
########################################################################################################################
################################################ Data acquisition ######################################################
print('2 done')
#   Train set (Original RGB image, Greyscale image)
nul_1, nul_2, train_patches = os.walk(os.path.join('datasets', network_name, 'philippe-plus-ISIC_2016_train_patches')).next()
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
    ####################### Parameters ########################
    ######################## Convolutional parameters

    # c_1
    c_1_weights = tf.get_variable("c_1_weights", shape=[5, 5, 3, 8], initializer=tf.contrib.layers.xavier_initializer())
    c_1_biases = tf.Variable(tf.zeros([8]), tf.float32)

    # c_2
    c_2_weights = tf.get_variable("c_2_weights", shape=[3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer())
    c_2_biases = tf.Variable(tf.zeros([16]), tf.float32)

    # c_3
    c_3_weights = tf.get_variable("c_3_weights", shape=[4, 4, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    c_3_biases = tf.Variable(tf.zeros([32]), tf.float32)

    # c_4
    c_4_weights = tf.get_variable("c_4_weights", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_4_biases = tf.Variable(tf.zeros([64]), tf.float32)

    # c_5
    c_5_weights = tf.get_variable("c_5_weights", shape=[5, 5, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    c_5_biases = tf.Variable(tf.zeros([64]), tf.float32)

    ###########################################################
    ######################## De-convolutional parameters

    # d_1
    d_1_weights = tf.get_variable("d_1_weights", shape=[5, 5, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    d_1_biases = tf.Variable(tf.zeros([64]), tf.float32)

    # d_2
    d_2_weights = tf.get_variable("d_2_weights", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    d_2_biases = tf.Variable(tf.zeros([32]), tf.float32)

    # d_3
    d_3_weights = tf.get_variable("d_3_weights", shape=[4, 4, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    d_3_biases = tf.Variable(tf.zeros([16]), tf.float32)

    # d_4
    d_4_weights = tf.get_variable("d_4_weights", shape=[3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer())
    d_4_biases = tf.Variable(tf.zeros([8]), tf.float32)

    # d_5
    d_5_weights = tf.get_variable("d_5_weights", shape=[5, 5, 1, 8], initializer=tf.contrib.layers.xavier_initializer())
    d_5_biases = tf.Variable(tf.zeros([1]), tf.float32)

    ###########################################################
    ########################### CNN ###########################
    ########################### Convolutional path

    # c_1 convolution
    c_1_filter = tf.nn.conv2d(x, c_1_weights, [1, 1, 1, 1], padding='VALID')
    c_1 = tf.nn.relu(c_1_filter + c_1_biases)
    '''c_1_beta = tf.Variable(tf.constant(0.0, shape=[c_1_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    c_1_gamma = tf.Variable(tf.constant(1.0, shape=[c_1_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    c_1_out = c_1_filter + c_1_biases
    c_1_batch_mean, c_1_batch_var = tf.nn.moments(c_1_out, [0])
    c_1 = tf.nn.relu(tf.nn.batch_normalization(c_1_out, c_1_batch_mean, c_1_batch_var, c_1_beta, c_1_gamma,variance_epsilon=1e-3))'''

    # c_2 convolution
    c_2_filter = tf.nn.conv2d(c_1, c_2_weights, [1, 1, 1, 1], padding='VALID')
    c_2 = tf.nn.relu(c_2_filter + c_2_biases)
    '''c_2_beta = tf.Variable(tf.constant(0.0, shape=[c_2_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    c_2_gamma = tf.Variable(tf.constant(1.0, shape=[c_2_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    c_2_out = c_2_filter + c_2_biases
    c_2_batch_mean, c_2_batch_var = tf.nn.moments(c_2_out, [0])
    c_2 = tf.nn.relu(tf.nn.batch_normalization(c_2_out, c_2_batch_mean, c_2_batch_var, c_2_beta, c_2_gamma, variance_epsilon=1e-3))'''

    # p_1 pooling
    p_1 = tf.nn.max_pool(c_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # c_3 convolution
    c_3_filter = tf.nn.conv2d(p_1, c_3_weights, [1, 1, 1, 1], padding='VALID')
    c_3 = tf.nn.relu(c_3_filter + c_3_biases)
    '''c_3_beta = tf.Variable(tf.constant(0.0, shape=[c_3_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    c_3_gamma = tf.Variable(tf.constant(1.0, shape=[c_3_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    c_3_out = c_3_filter + c_3_biases
    c_3_batch_mean, c_3_batch_var = tf.nn.moments(c_3_out, [0])
    c_3 = tf.nn.relu(tf.nn.batch_normalization(c_3_out, c_3_batch_mean, c_3_batch_var, c_3_beta, c_3_gamma, variance_epsilon=1e-3))'''


    # p_2 pooling
    p_2 = tf.nn.max_pool(c_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # c_4 convolution
    c_4_filter = tf.nn.conv2d(p_2, c_4_weights, [1, 1, 1, 1], padding='VALID')
    c_4 = tf.nn.relu(c_4_filter + c_4_biases)
    '''c_4_beta = tf.Variable(tf.constant(0.0, shape=[c_4_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    c_4_gamma = tf.Variable(tf.constant(1.0, shape=[c_4_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    c_4_out = c_4_filter + c_4_biases
    c_4_batch_mean, c_4_batch_var = tf.nn.moments(c_4_out, [0])
    c_4 = tf.nn.relu(tf.nn.batch_normalization(c_4_out, c_4_batch_mean, c_4_batch_var, c_4_beta, c_4_gamma, variance_epsilon=1e-3))'''

    # p_3 pooling
    p_3 = tf.nn.max_pool(c_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # c_5 convolution
    c_5_filter = tf.nn.conv2d(p_3, c_5_weights, [1, 1, 1, 1], padding='VALID')
    c_5 = tf.nn.relu(c_5_filter + c_5_biases)
    '''c_5_beta = tf.Variable(tf.constant(0.0, shape=[c_5_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    c_5_gamma = tf.Variable(tf.constant(1.0, shape=[c_5_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    c_5_out = c_5_filter + c_5_biases
    c_5_batch_mean, c_5_batch_var = tf.nn.moments(c_5_out, [0])
    c_5 = tf.nn.relu(tf.nn.batch_normalization(c_5_out, c_5_batch_mean, c_5_batch_var, c_5_beta, c_5_gamma, variance_epsilon=1e-3))'''

    ###########################################################
    ########################### De-convolutional path

    # d_1 deconvolutional
    d_1_filter = tf.nn.conv2d_transpose(c_5, d_1_weights, [batch_size, c_5.get_shape().as_list()[1] + 4, c_5.get_shape().as_list()[2] + 4, 64], [1, 1, 1, 1], padding='VALID', name=None)
    d_1 = tf.nn.relu(d_1_filter + d_1_biases)+p_3
    '''d_1_beta = tf.Variable(tf.constant(0.0, shape=[d_1_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    d_1_gamma = tf.Variable(tf.constant(1.0, shape=[d_1_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    d_1_out = d_1_filter + d_1_biases
    d_1_batch_mean, d_1_batch_var = tf.nn.moments(d_1_out, [0])
    d_1 = tf.nn.relu(tf.nn.batch_normalization(d_1_out, d_1_batch_mean, d_1_batch_var, d_1_beta, d_1_gamma, variance_epsilon=1e-3))'''

    # u_1 up-sampling
    u_1 = tf.image.resize_bilinear(d_1, (d_1.get_shape().as_list()[1] * 2, d_1.get_shape().as_list()[2] * 2))+c_4

    # d_2 deconvolutional
    d_2_filter = tf.nn.conv2d_transpose(u_1, d_2_weights, [batch_size, u_1.get_shape().as_list()[1] + 3, u_1.get_shape().as_list()[2] + 3, 32], [1, 1, 1, 1], padding='VALID', name=None)
    d_2 = tf.nn.relu(d_2_filter + d_2_biases)+p_2
    '''d_2_beta = tf.Variable(tf.constant(0.0, shape=[d_2_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    d_2_gamma = tf.Variable(tf.constant(1.0, shape=[d_2_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    d_2_out = d_2_filter + d_2_biases
    d_2_batch_mean, d_2_batch_var = tf.nn.moments(d_2_out, [0])
    d_2 = tf.nn.relu(tf.nn.batch_normalization(d_2_out, d_2_batch_mean, d_2_batch_var, d_2_beta, d_2_gamma, variance_epsilon=1e-3))'''

    # u_2 up-sampling
    u_2 = tf.image.resize_bilinear(d_2, (d_2.get_shape().as_list()[1] * 2, d_2.get_shape().as_list()[2] * 2))+c_3

    # d_3 deconvolutional
    d_3_filter = tf.nn.conv2d_transpose(u_2, d_3_weights, [batch_size, u_2.get_shape().as_list()[1] + 3, u_2.get_shape().as_list()[2] + 3, 16], [1, 1, 1, 1], padding='VALID', name=None)
    d_3 = tf.nn.relu(d_3_filter + d_3_biases)+p_1
    '''d_3_beta = tf.Variable(tf.constant(0.0, shape=[d_3_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    d_3_gamma = tf.Variable(tf.constant(1.0, shape=[d_3_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    d_3_out = d_3_filter + d_3_biases
    d_3_batch_mean, d_3_batch_var = tf.nn.moments(d_3_out, [0])
    d_3 = tf.nn.relu(tf.nn.batch_normalization(d_3_out, d_3_batch_mean, d_3_batch_var, d_3_beta, d_3_gamma, variance_epsilon=1e-3))'''

    # u_3 up-sampling
    u_3 = tf.image.resize_bilinear(d_3, (d_3.get_shape().as_list()[1] * 2, d_3.get_shape().as_list()[2] * 2))+c_2

    # d_4 deconvolutional
    d_4_filter = tf.nn.conv2d_transpose(u_3, d_4_weights, [batch_size, u_3.get_shape().as_list()[1] + 2, u_3.get_shape().as_list()[2] + 2, 8], [1, 1, 1, 1], padding='VALID', name=None)
    d_4 = tf.nn.relu(d_4_filter + d_4_biases)+c_1
    '''d_4_beta = tf.Variable(tf.constant(0.0, shape=[d_4_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    d_4_gamma = tf.Variable(tf.constant(1.0, shape=[d_4_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    d_4_out = d_4_filter + d_4_biases
    d_4_batch_mean, d_4_batch_var = tf.nn.moments(d_4_out, [0])
    d_4 = tf.nn.relu(tf.nn.batch_normalization(d_4_out, d_4_batch_mean, d_4_batch_var, d_4_beta, d_4_gamma, variance_epsilon=1e-3))'''

    # d_5 deconvolutional
    d_5_filter = tf.nn.conv2d_transpose(d_4, d_5_weights, [batch_size, d_4.get_shape().as_list()[1] + 4, d_4.get_shape().as_list()[2] + 4, 1], [1, 1, 1, 1], padding='VALID', name=None)
    d_5 = tf.nn.sigmoid(d_5_filter + d_5_biases, name="estimated_segmentation")
    '''d_5_beta = tf.Variable(tf.constant(0.0, shape=[d_5_filter.get_shape().as_list()[3]]), name='beta', trainable=True)
    d_5_gamma = tf.Variable(tf.constant(1.0, shape=[d_5_filter.get_shape().as_list()[3]]), name='gamma', trainable=True)
    d_5_out = d_5_filter + d_5_biases
    d_5_batch_mean, d_5_batch_var = tf.nn.moments(d_5_out, [0])
    d_5 = tf.nn.sigmoid(tf.nn.batch_normalization(d_5_out, d_5_batch_mean, d_5_batch_var, d_5_beta, d_5_gamma, variance_epsilon=1e-3), name="estimated_segmentation")'''

    ###########################################################
    ########################### Loss function
    X_L = tf.squeeze(x_labels)
    D_5 = tf.squeeze(d_5)
    Num = tf.reduce_sum(X_L * D_5)
    De_num_1 = tf.reduce_sum(X_L * X_L)
    De_num_2 = tf.reduce_sum(D_5 * D_5)
    loss = 1 - (Num/(De_num_1+De_num_2-Num))
    #loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_labels, logits=d_5))
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

            _, l, C_1_W, C_2_W, C_3_W, C_4_W, C_5_W, D_1_W, D_2_W, D_3_W, D_4_W, D_5_W, C_1_B, C_2_B, C_3_B, C_4_B, C_5_B, D_1_B, D_2_B, D_3_B, D_4_B, D_5_B = sess.run(
                [optimizer, loss,
                 c_1_weights, c_2_weights, c_3_weights, c_4_weights, c_5_weights,
                 d_1_weights, d_2_weights, d_3_weights, d_4_weights, d_5_weights,
                 c_1_biases, c_2_biases, c_3_biases, c_4_biases, c_5_biases,
                 d_1_biases, d_2_biases, d_3_biases, d_4_biases, d_5_biases],
                {x: train_set_ref, x_labels: train_set_ref_label})
            print("epoch ", epoch, "      iteration ", count, "      loss ", l)
            Loss_avg_iter_train.append(l)
        if '-save' in sys.argv:
            with open(os.path.join(path_results, 'Loss_avg_iter_train.txt'), 'a') as file:
                file.writelines('\n%s' %(Loss_avg_iter_train))

        if '-save' in sys.argv:
            if epoch%1 == 0:
                print('saving !')
                np.save(os.path.join(path_results, "c_1_weights"), C_1_W)
                np.save(os.path.join(path_results, "c_2_weights"), C_2_W)
                np.save(os.path.join(path_results, "c_3_weights"), C_3_W)
                np.save(os.path.join(path_results, "c_4_weights"), C_4_W)
                np.save(os.path.join(path_results, "c_5_weights"), C_5_W)

                np.save(os.path.join(path_results, "d_1_weights"), D_1_W)
                np.save(os.path.join(path_results, "d_2_weights"), D_2_W)
                np.save(os.path.join(path_results, "d_3_weights"), D_3_W)
                np.save(os.path.join(path_results, "d_4_weights"), D_4_W)
                np.save(os.path.join(path_results, "d_5_weights"), D_5_W)

                np.save(os.path.join(path_results, "c_1_biases"), C_1_B)
                np.save(os.path.join(path_results, "c_2_biases"), C_2_B)
                np.save(os.path.join(path_results, "c_3_biases"), C_3_B)
                np.save(os.path.join(path_results, "c_4_biases"), C_4_B)
                np.save(os.path.join(path_results, "c_5_biases"), C_5_B)

                np.save(os.path.join(path_results, "d_1_biases"), D_1_B)
                np.save(os.path.join(path_results, "d_2_biases"), D_2_B)
                np.save(os.path.join(path_results, "d_3_biases"), D_3_B)
                np.save(os.path.join(path_results, "d_4_biases"), D_4_B)
                np.save(os.path.join(path_results, "d_5_biases"), D_5_B)
print('hello')
############################################################################################










