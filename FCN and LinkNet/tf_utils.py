import tensorflow as tf


def tf_reshape_patch_vec_in_mat(tf_patch_vec, patch_size):
    """

    :param tf_patch_vec: Tensor, shape = [num_batches, in_height * in_width]
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: tf_patch_mat: Tensor, shape = [num_batches, in_height, in_width]
    """
    if tf_patch_vec.get_shape().ndims is 1:
        raise ValueError('Vectorized patch must be of shape [num_batches, in_height*in_width]')
    elif tf_patch_vec.get_shape().ndims > 2:
        # TODO: make patches [num_batches, in_height*in_width, in_channels]
        raise NotImplementedError('Only supports vectorized patch of shape [num_batches, in_height*in_width]')
    else:
        patch_vec_shape = tf_patch_vec.get_shape().as_list()
        tf_patch_mat = tf.reshape(tf_patch_vec, shape=[patch_vec_shape[0], patch_size[0], patch_size[1]])

    return tf_patch_mat


def tf_reshape_patch_mat_in_vec(tf_patch_mat):
    """

    :param tf_patch_mat: Tensor, shape = [num_batches, in_height, in_width]
    :return: tf_patch_vec: Tensor, shape = [num_batches, in_height * in_width]
    """
    if tf_patch_mat.get_shape().ndims is 1:
        raise ValueError('Patch must be of shape [num_batches, in_height, in_width]')
    elif tf_patch_mat.get_shape().ndims > 3:
        # TODO: make patches [num_batches, in_height, in_width, in_channels]
        raise NotImplementedError('Only supports patch of shape [num_batches, in_height, in_width]')
    else:
        patch_mat_shape = tf_patch_mat.get_shape().as_list()
        tf_patch_vec = tf.reshape(tf_patch_mat, shape=[patch_mat_shape[0], patch_mat_shape[1]*patch_mat_shape[2]])

    return tf_patch_vec


def tf_soft_thresholding(tf_coeff, tf_theta):
    """
    Soft thresholding
    :param tf_coeff: Tensor, shape = [batch_size, coeff_number]
    :param tf_theta: Tensor, shape = [batch_size, coeff_number] OR a scalar Tensor, shape=()
    :return: coeff_th: Tensor, shape = [batch_size, coeff_number]
    """
    # TODO: check on the shape of coeff: can ben (coeff_number, ) or (coeff_number, 1)
    # tf_theta = tf.maximum(tf.constant(0, dtype=tf.float32), tf_theta)  # CANNOT BE <0
    tf_coeff_th = tf.nn.relu(tf_coeff - tf_theta) - tf.nn.relu(-tf_coeff - tf_theta)

    return tf_coeff_th