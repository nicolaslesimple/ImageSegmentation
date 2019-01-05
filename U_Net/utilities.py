import tensorflow as tf
import numpy as np


def weight_def(shape, stddev = 0.1, name = 'undefined'):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name = name)


def bias_def(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name = name)


def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def conv2d_dropout_relu(x, W, B, n_out, keep_prob, phase_train, name):
    bn = batch_norm(x, n_out, phase_train, parent_name = name)
    conv2d = tf.nn.conv2d(bn, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
    dropout = tf.nn.dropout(conv2d, keep_prob)
    return tf.nn.relu(tf.nn.bias_add(dropout, B))


def maxpool2d(x, name = 'undefined'):
    """ ksize defines the size of the pool window
     strides defines the way the window moves (movement of the window)
     """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)



def deconv2d_relu(x, W, B, n_out,  phase_train, name):

    bn = batch_norm(x, n_out, phase_train, parent_name = name)

    #stride = upscale_factor #upscale_factor
    stride = 2
    strides = [1, stride, stride, 1]
    # Shape of the x tensor
    in_shape = tf.shape(bn)

    h = ((in_shape[1] - 1) * stride) + 2 - (in_shape[1]*2%5) #- 1*(bool(in_shape[1] < 20))#
    w = ((in_shape[2] - 1) * stride) + 2 - (in_shape[2]*2%5) #- 1*(bool(in_shape[2] < 20))#(in_shape[2]%2)
    new_shape = [in_shape[0], h, w, W.shape[3]]
    output_shape = tf.stack(new_shape)
    deconv = tf.nn.conv2d_transpose(bn, W, output_shape,
                                    strides=strides, padding='SAME', name = name)
    return tf.nn.relu(tf.nn.bias_add(deconv, B))



# Unused because buggy
def crop_and_concat(x1, x2, name):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)



# Copy pasted functions
def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")





def accuracy(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
                (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))






def batch_norm(x, n_out, phase_train, parent_name):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta_' + parent_name, trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma_' + parent_name, trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,\
                            mean_var_with_update,\
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed


'''
def write_submission(merged_pred_array, save_path, threshold):
    """Converts images into a submission file"""
    with open(save_path, 'w') as f:
        f.write('id,prediction\n')
        for index, img in enumerate(merged_pred_array):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(img, index+1, threshold))'''


# assign a label to a patch
def patch_to_label(patch, threshold):
    foreground_threshold = threshold # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

'''
def mask_to_submission_strings(img, img_number, threshold):
    """Reads a single image and outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, threshold)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))'''
