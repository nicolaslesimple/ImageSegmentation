import os
import shutil
import warnings
import zipfile
from itertools import product

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import scipy.linalg
import scipy.misc
from scipy import signal
from scipy.ndimage.filters import convolve
import re
from PIL import Image


def is_array_str(obj):
    """
    Check if obj is a list of strings or a tuple of strings or a set of strings
    :param obj: an object
    :return: flag: True or False
    """
    # TODO: modify the use of is_array_str(obj) in the code to is_array_of(obj, classinfo)
    flag = False
    if isinstance(obj, str):
        pass
    elif all(isinstance(item, str) for item in obj):
        flag = True

    return flag


def is_array_of(obj, classinfo):
    """
    Check if obj is a list of classinfo or a tuple of classinfo or a set of classinfo
    :param obj: an object
    :param classinfo: type of class (or subclass). See isinstance() build in function for more info
    :return: flag: True or False
    """
    flag = False
    if isinstance(obj, classinfo):
        pass
    elif all(isinstance(item, classinfo) for item in obj):
        flag = True

    return flag


def check_and_convert_to_list_str(obj):
    """
    Check if obj is a string or an array like of strings and return a list of strings
    :param obj: and object
    :return: list_str: a list of strings
    """
    if isinstance(obj, str):
        list_str = [obj]  # put in a list to avoid iterating on characters
    elif is_array_str(obj):
        list_str = []
        for item in obj:
            list_str.append(item)
    else:
        raise TypeError('Input must be a string or an array like of strings.')

    return list_str

def load_results(path, file_ext=''):
    """
    Load images in grayscale from the path
    :param path: path to folder
    :param file_ext: a string or a list of strings (even an array like of strings)
    :return: image_list, image_name_list
    """
    # Check file_ext type
    file_ext = check_and_convert_to_list_str(file_ext)

    file_list = []
    file_name_list = []
    for file in os.listdir(path):
        file_name, ext = os.path.splitext(file)
        if ext.lower() not in file_ext:
            continue
        file_list.append(np.load(os.path.join(path, file)))
        file_name_list.append(file_name)

    return file_list, file_name_list

def load_images(path, file_ext='.png'):
    """
    Load images in grayscale from the path
    :param path: path to folder
    :param file_ext: a string or a list of strings (even an array like of strings)
    :return: image_list, image_name_list
    """
    # Check file_ext type
    file_ext = check_and_convert_to_list_str(file_ext)

    image_list = []
    image_name_list = []
    for file in os.listdir(path):
        file_name, ext = os.path.splitext(file)
        if ext.lower() not in file_ext:
            continue
        # Import image and convert it to 8-bit pixels, black and white (using mode='L')
        image_list.append(scipy.misc.imread(os.path.join(path, file), mode='L'))
        image_name_list.append(file_name)

    return image_list, image_name_list


def extract_zip_archive(zip_file_path, extract_path, file_ext=''):
    """
    Extract zip archive. If file_ext is specified, only extracts files with specified extension
    :param zip_file_path: path to zip archive
    :param extract_path: path to export folder
    :param file_ext: a string or a list of strings (even an array like of strings)
    :return:
    """
    # Check file_ext type
    file_ext = check_and_convert_to_list_str(file_ext)

    # Check if export_path already contains the files with a valid extension
    valid_files_in_extract_path = [os.path.join(root, name)
                                   for root, dirs, files in os.walk(extract_path)
                                   for name in files
                                   if name.endswith(tuple(file_ext))]

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        files_to_extract = [name for name in zip_ref.namelist()
                            if name.endswith(tuple(file_ext))
                            and os.path.join(extract_path, name) not in valid_files_in_extract_path]

        # Only extracts files if not already extracted
        # TODO: load directly the images without extracting them
        for file in files_to_extract:
            print(file)
            zip_ref.extract(file, path=extract_path)

    return


def export_image_list(image_list, image_name_list, export_name_list=True, path='', file_ext='.png'):
    """
    Export images export_name_list of (image_list, image_name_list) in path as image_name.ext
    :param image_list: list of array
    :param image_name_list: list of strings
    :param export_name_list: True, False, None, string or list of strings
    :param path: path to export folder
    :param file_ext: file extension
    :return:
    """
    # Check if file_ext is a string
    if not isinstance(file_ext, str):
        raise TypeError('File extension must be a string')

    # Check if image_name_list is list of string or simple string
    image_name_list = check_and_convert_to_list_str(image_name_list)

    # Check export_name_list type
    #   if is True, i.e. will export all images
    #   Otherwise check if export_name_list is list of strings or simple string
    if isinstance(export_name_list, bool):
        if export_name_list:  # True case
            export_name_list = image_name_list
        else:  # False case
            export_name_list = ['']
    elif export_name_list is None:
        export_name_list = ['']
    else:
        export_name_list = check_and_convert_to_list_str(export_name_list)

    # Check if folder already exists
    if os.path.exists(path):  # never True if path = ''
        # Check if folder content is exactly the same as what will be exported
        if not sorted(os.listdir(path)) == [item + file_ext for item in sorted(export_name_list)]:
            shutil.rmtree(path)
            print('Folder {} has been removed'.format(path))
        else:
            return

    # Check if folder doesn't exist and if path not empty to create the folder
    if not os.path.exists(path) and path:
        os.makedirs(path)
        print('Folder {} has been created'.format(path))

    # Save images
    for i, image_name in enumerate(image_name_list):
        if image_name not in export_name_list:
            continue
        scipy.misc.imsave(os.path.join(path, image_name + file_ext), image_list[i])
        print('Saved {} {} as {}'.format(image_name, image_list[i].shape, os.path.join(path, image_name + file_ext)))

    return


def get_data_paths(dataset_name):
    """
    Generate and return data paths
    :param dataset_name: string
    :return: data_paths: dict
    """
    if not isinstance(dataset_name, str):
        raise TypeError('Data set name must be a string')

    keys = ['sources_base', 'source', 'source_archive', 'dataset', 'orig', 'train', 'test', 'valid']
    data_paths = dict.fromkeys(keys)

    data_paths['sources_base'] = os.path.join('datasets', 'sources')
    data_paths['source'] = os.path.join(data_paths['sources_base'], dataset_name)
    data_paths['source_archive'] = data_paths['source'] + '.zip'
    data_paths['dataset'] = os.path.join('datasets', dataset_name)
    data_paths['orig'] = os.path.join(data_paths['dataset'], 'orig')
    data_paths['train'] = os.path.join(data_paths['dataset'], 'train')
    data_paths['test'] = os.path.join(data_paths['dataset'], 'test')
    data_paths['valid'] = os.path.join(data_paths['dataset'], 'valid')

    return data_paths


def generate_original_images(dataset_name):
    """
    Generate original images
    :param dataset_name: name of the dataset such that dataset.zip exists
    :return:
    """
    # TODO: download from the web so it doesn't have to be hosted on github

    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp', '.pgm']
    export_path = os.path.join('datasets', dataset_name, 'orig')

    # Unzip archive
    extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    # Loading valid image in grayscale
    image_list, image_name_list = load_images(data_source_path, file_ext=valid_ext)

    # Export original images
    export_image_list(image_list, image_name_list, path=export_path, file_ext='.png')

    return


def export_set_from_orig(dataset_name, set_name, name_list):
    """
    Export a set from the original set based on the name list provided
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param set_name: string, name of the set (yet only 'train' and 'test')
    :param name_list: image name list to extract from the 'orig' set
    :return:
    """
    # Get paths
    data_paths = get_data_paths(dataset_name)

    # Load original images
    orig_image_list, orig_name_list = load_images(data_paths['orig'], file_ext='.png')

    export_image_list(orig_image_list, orig_name_list, export_name_list=name_list,
                      path=data_paths[set_name], file_ext='.png')

    return

def generate_train_images_patches_v2(dataset_name, patch_size=(1,1)):
    """
    Generate training image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """

    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp', '.pgm']
    export_path = os.path.join('datasets', 'network_v2', 'train_patches')

    # Unzip archive
    extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    k = 0
    for file in os.listdir(data_source_path):
        file_name, ext = os.path.splitext(file)
        # Import image and convert it to 8-bit pixels, black and white (using mode='L')
        image = scipy.misc.imread(os.path.join(data_source_path, file), mode='L')
        image_name = file_name
        if image.shape[0] >= patch_size[0] and image.shape[1] >= patch_size[1]:
            image_patch = reshape_patch_in_vec(extract_2d_patches(image, patch_size))
            for i in range(image_patch.shape[0]):
                image_patch_path = os.path.join(export_path, str(k))
                np.save(image_patch_path, image_patch[i,:])
                k = k + 1

    return


def generate_train_images_patches_saleh(dataset_name, patch_size=(32,32), number_files=1, batch_size=100, number_patches_total = 500):
    """
    Generate training image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """

    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    #data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp', '.pgm','.jpeg']

    for file_ind in range(number_files):
        if not os.path.exists(os.path.join('datasets', 'network', 'train_patches_p') + str(file_ind+1)):
            os.makedirs(os.path.join('datasets', 'network', 'train_patches_p') + str(file_ind+1))
        else:
            shutil.rmtree(os.path.join('datasets', 'network', 'train_patches_p') + str(file_ind+1))
            os.makedirs(os.path.join('datasets', 'network', 'train_patches_p') + str(file_ind+1))


        #os.makedirs(os.path.join('datasets', 'network', 'train_patches_48_p') + str(file_ind+1))

    # Unzip archive
    #extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    k = 0
    jj = 0
    for file in os.listdir(data_source_path):
        file_name, ext = os.path.splitext(file)
        # Import image and convert it to 8-bit pixels, black and white (using mode='L')
        image = scipy.misc.imread(os.path.join(data_source_path, file), mode='L')
        image_name = file_name
        if image.shape[0] >= patch_size[0] and image.shape[1] >= patch_size[1]:
            image_patch = reshape_patch_in_vec(extract_2d_patches(image, patch_size))
            for i in range(image_patch.shape[0]):
                if k < number_patches_total:
                    print(k)
                    save_ind = int(np.true_divide(k,batch_size))
                    image_patch_path = os.path.join(os.path.join('datasets', 'network', 'train_patches_p') + str(save_ind+1), str(jj))
                    np.save(image_patch_path, image_patch[i, :])
                    k = k + 1
                    jj = jj + 1
                    if jj >= batch_size:
                        jj = 0

    return

def generate_train_images_patches_saleh_DermoSafe(dataset_name, color, image_width, image_height):
    """
    Generate training image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """
    # Parameters
    data_sources_main_path = os.path.join('datasets', 'network')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)

    shutil.rmtree(os.path.join('datasets', 'network', 'valid'))
    os.makedirs(os.path.join('datasets', 'network', 'valid'))

    #indice = 40000
    for file in os.listdir(data_source_path):
        file_name, ext = os.path.splitext(file)
        indice = int(re.findall(r'\d+', file_name)[0])
        if color == 'greyscale':
            image = scipy.misc.imread(os.path.join(data_source_path, file), mode='L')
        if color == 'RGB':
            image = scipy.misc.imread(os.path.join(data_source_path, file), mode='RGB')
        if image.shape[0]/image.shape[1] != 0.75:
            if color == 'greyscale':
                image = Image.fromarray(image, 'L')
            if color == 'RGB':
                image = Image.fromarray(image, 'RGB')

            width = image.size[0]
            height = image.size[1]

            aspect = width / float(height)

            ideal_width = image_width
            ideal_height = image_height

            ideal_aspect = ideal_width / float(ideal_height)

            if aspect > ideal_aspect:
                # Then crop the left and right edges:
                new_width = int(ideal_aspect * height)
                offset = (width - new_width) / 2
                resize = (offset, 0, width - offset, height)
            else:
                # ... crop the top and bottom:
                new_height = int(width / ideal_aspect)
                offset = (height - new_height) / 2
                resize = (0, offset, width, height - offset)

            thumb = image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)
            image = np.asarray(thumb)

        print('ratio = ', image.shape[0]/image.shape[1])
        #image = scipy.misc.imresize(image, (512, 512), interp='bilinear', mode=None)
        image_patch_path = os.path.join(os.path.join('datasets', 'network', 'valid'), str(indice)+str(900))
        scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(image_patch_path + '.png')
        #indice = indice + 1
    return

def generate_train_images_patches(dataset_name, patch_size=(32,32)):
    """
    Generate training image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """

    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp', '.pgm']
    export_path = os.path.join('datasets', 'network', 'train_patches_48')

    # Unzip archive
    extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    k = 0
    for file in os.listdir(data_source_path):
        file_name, ext = os.path.splitext(file)
        # Import image and convert it to 8-bit pixels, black and white (using mode='L')
        image = scipy.misc.imread(os.path.join(data_source_path, file), mode='L')
        image_name = file_name
        if image.shape[0] >= patch_size[0] and image.shape[1] >= patch_size[1]:
            image_patch = reshape_patch_in_vec(extract_2d_patches(image, patch_size))
            for i in range(image_patch.shape[0]):
                image_patch_path = os.path.join(export_path, str(k))
                np.save(image_patch_path, image_patch[i,:])
                k = k + 1

    return

def generate_valid_images(dataset_name, name_list=None):
    """
    Generate training image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """

    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp', '.pgm']
    export_path = os.path.join('datasets', 'network', 'valid')

    # Unzip archive
    extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    # Loading valid image in grayscale
    image_list, image_name_list = load_images(data_source_path, file_ext=valid_ext)

    # Export original images
    export_image_list(image_list, image_name_list, path=export_path, file_ext='.png')

    return


def generate_test_images(dataset_name, name_list=None):
    """
    Generate testing image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """
    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp', '.pgm']
    export_path = os.path.join('datasets', 'network', 'test')

    # Unzip archive
    extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    # Loading valid image in grayscale
    image_list, image_name_list = load_images(data_source_path, file_ext=valid_ext)

    # Export original images
    export_image_list(image_list, image_name_list, path=export_path, file_ext='.png')

    return


def extract_2d_patches_rgb_image(image, patch_size):
    """
    Extract non-overlapping patches of size patch_height x patch_width
    :param image: array, shape = (image_height, image_width)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: patches: array, shape = (patches_number, patch_height, patch_width)
    """
    image_size = np.asarray([image.shape[0], image.shape[1]])  # convert to numpy array to allow array computations
    patch_size = np.asarray(patch_size)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    patches = np.zeros([np.prod(patches_number), patch_size[0], patch_size[1], 3])
    # patches = np.zeros([patch_size[0], patch_size[1], np.prod(patches_number)])

    # Cartesian iteration using itertools.product()
    # Equivalent to the nested for loop
    # for r in range(patches_number[0]):
    #     for c in range(patches_number[1]):
    for k, (r, c) in zip(range(np.prod(patches_number)), product(range(patches_number[0]), range(patches_number[1]))):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        patches[k, :, :, :] += image[rr:rr + patch_size[0], cc:cc + patch_size[1], :]
        # patches[:, :, k] += image[rr:rr + patch_size[0], cc:cc + patch_size[1]]  # TODO: use [..., k]

    return patches


def extract_2d_patches(image, patch_size):
    """
    Extract non-overlapping patches of size patch_height x patch_width
    :param image: array, shape = (image_height, image_width)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: patches: array, shape = (patches_number, patch_height, patch_width)
    """
    image_size = np.asarray(image.shape)  # convert to numpy array to allow array computations
    patch_size = np.asarray(patch_size)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    patches = np.zeros([np.prod(patches_number), patch_size[0], patch_size[1]])
    # patches = np.zeros([patch_size[0], patch_size[1], np.prod(patches_number)])

    # Cartesian iteration using itertools.product()
    # Equivalent to the nested for loop
    # for r in range(patches_number[0]):
    #     for c in range(patches_number[1]):
    for k, (r, c) in zip(range(np.prod(patches_number)), product(range(patches_number[0]), range(patches_number[1]))):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        patches[k, :, :] += image[rr:rr + patch_size[0], cc:cc + patch_size[1]]
        # patches[:, :, k] += image[rr:rr + patch_size[0], cc:cc + patch_size[1]]  # TODO: use [..., k]

    return patches


def reconstruct_from_2d_patches(patches, image_size):
    """
    Reconstruct image from patches of size patch_height x patch_width
    :param patches: array, shape = (patches_number, patch_height, patch_width)
    :param image_size: tuple of ints (image_height, image_width)
    :return: rec_image: array of shape (rec_image_height, rec_image_width)
    """
    image_size = np.asarray(image_size)         # convert to numpy array to allow array computations
    patch_size = np.asarray(patches[0].shape)   # convert to numpy array to allow array computations
    # patch_size = np.asarray(patches[:, :, 0].shape)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    rec_image_size = patches_number * patch_size
    rec_image = np.zeros(rec_image_size)

    # Cartesian iteration using itertools.product()
    for k, (r, c) in zip(range(np.prod(patches_number)), product(range(patches_number[0]), range(patches_number[1]))):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        rec_image[rr:rr + patch_size[0], cc:cc + patch_size[1]] += patches[k, :, :]
        # rec_image[rr:rr + patch_size[0], cc:cc + patch_size[1]] += patches[:, :, k]  # TODO: use [..., k]

    return rec_image


def reshape_patch_in_vec(patches):
    # """
    # :param patches: array, shape = (patch_height, patch_width, patches_number)
    # :return: vec_patches: array, shape = (patch_height * patch_width, patches_number)
    # """
    # # Check if only a single patch (i.e. ndim = 2) or multiple patches (i.e. ndim = 3)
    # if patches.ndim == 2:
    #     vec_patches = patches.reshape((patches.shape[0]*patches.shape[1]))
    # elif patches.ndim == 3:
    #     vec_patches = patches.reshape((patches.shape[0]*patches.shape[1], patches.shape[-1]))
    # else:
    #     raise TypeError('Patches cannot have more than 3 dimensions (i.e. only grayscale for now)')
    #
    # return vec_patches

    """
    :param patches: array, shape = (patches_number, patch_height, patch_width)
    :return: vec_patches: array, shape = (patches_number, patch_height * patch_width)
    """
    # Check if only a single patch (i.e. ndim = 2) or multiple patches (i.e. ndim = 3)
    if patches.ndim == 2:
        vec_patches = patches.reshape((patches.shape[0] * patches.shape[1]))
    elif patches.ndim == 3:
        vec_patches = patches.reshape((patches.shape[0], patches.shape[1] * patches.shape[2]))
    else:
        raise TypeError('Patches cannot have more than 3 dimensions (i.e. only grayscale for now)')

    return vec_patches


def reshape_vec_in_patch(vec_patches, patch_size):
    # """
    # :param vec_patches: array, shape = (patch_height * patch_width, patches_number)
    # :param patch_size: tuple of ints (patch_height, patch_width)
    # :return patches: array, shape = (patch_height, patch_width, patches_number)
    # """
    # # Check if vec_patches is 1D (i.e. only one patch) or 2D (i.e. multiple patches)
    # if vec_patches.ndim == 1:
    #     patches = vec_patches.reshape((patch_size[0], patch_size[1]))
    # elif vec_patches.ndim == 2:
    #     patches = vec_patches.reshape((patch_size[0], patch_size[1], vec_patches.shape[-1]))
    # else:
    #     raise TypeError('Vectorized patches array cannot be more than 2D')
    #
    # return patches

    """
    :param vec_patches: array, shape = (patches_number, patch_height * patch_width)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return patches: array, shape = (patches_number, patch_height, patch_width)
    """
    # Check if vec_patches is 1D (i.e. only one patch) or 2D (i.e. multiple patches)
    if vec_patches.ndim == 1:
        patches = vec_patches.reshape((patch_size[0], patch_size[1]))
    elif vec_patches.ndim == 2:
        patches = vec_patches.reshape((vec_patches.shape[0], patch_size[0], patch_size[1]))
    else:
        raise TypeError('Vectorized patches array cannot be more than 2D')

    return patches


def generate_vec_set(image_list, patch_size):
    """
    Generate vectorized set of image based on patch_size
    :param image_list: list of array
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: vec_set: array, shape = (patches_number, patch_height * patch_width)
    """
    # TODO: check if image_list is a list!
    if not isinstance(image_list, list):
        raise NotImplementedError('Must be a list, implement the case of single image')

    patch_list = []
    for _, image in enumerate(image_list):
        patch_list.append(extract_2d_patches(image, patch_size))

    patches = np.concatenate(patch_list, axis=0)
    # patches = np.concatenate(patch_list, axis=-1)

    vec_set = reshape_patch_in_vec(patches)

    return vec_set


def generate_cross_validation_sets(full_set, fold_number=5, fold_combination=1):
    """
    Generate cross validations sets (i.e train and validation sets) w.r.t. a total fold number and the fold combination
    :param full_set: array, shape = (set_size, set_dim)
    :param fold_number: positive int
    :param fold_combination: int
    :return: train_set, val_set
    """
    if not isinstance(fold_combination, int):
        raise TypeError('Fold combination must be an integer')
    if not isinstance(fold_number, int):
        raise TypeError('Fold number must be an integer')
    if fold_number < 1:
        raise ValueError('Fold number must be a positive integer')
    if fold_combination > fold_number:
        raise ValueError('Fold combination must be smaller or equal to fold number')
    if not isinstance(full_set, np.ndarray):
        raise TypeError('Full set must be a numpy array')
    if full_set.ndim is not 2:
        raise TypeError('Full set must be a 2 dimensional array')

    # patch_number = full_set.shape[1]
    # fold_len = int(patch_number / fold_number)  # int -> floor
    # val_set_start = (fold_combination - 1) * fold_len
    # val_set_range = range(val_set_start, val_set_start + fold_len)
    # train_set_list = [idx for idx in range(fold_number * fold_len) if idx not in val_set_range]
    # train_set = full_set[:, train_set_list]
    # val_set = full_set[:, val_set_range]
    patch_number = full_set.shape[0]
    fold_len = int(patch_number / fold_number)  # int -> floor
    val_set_start = (fold_combination - 1) * fold_len
    val_set_range = range(val_set_start, val_set_start + fold_len)
    train_set_list = [idx for idx in range(fold_number * fold_len) if idx not in val_set_range]
    train_set = full_set[train_set_list, :]     # TODO: use [train_set_list] or [train_set_list, ...]
    val_set = full_set[val_set_range, :]        # TODO: use [val_set_range] or [val_set_range, ...]

    return train_set, val_set


def create_gaussian_rip_matrix(size=None, seed=None):
    """
    Create a Gaussian matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional. Default is None
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    mean = 0.0
    stdev = 1 / np.sqrt(m)
    prng = np.random.RandomState(seed=seed)
    matrix = prng.normal(loc=mean, scale=stdev, size=size)

    return matrix


def create_gaussian_orth_matrix(size=None, seed=None):
    """
    Create an orthonormalized Gaussian matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional. Default is None
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    mean = 0.0
    stdev = 1 / np.sqrt(m)
    prng = np.random.RandomState(seed=seed)
    matrix_orth = prng.normal(loc=mean, scale=stdev, size=(n, n))
    matrix_orth = scipy.linalg.orth(matrix_orth)
    matrix = matrix_orth[0:m, :]

    return matrix


def create_bernoulli_rip_matrix(size=None, seed=None):
    """
    Create a Bernoulli matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional. Default is None
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    prng = np.random.RandomState(seed=seed)
    matrix = prng.randint(low=0, high=2, size=size).astype('float')  # gen 0, +1 sequence
    # astype('float') required to use the true divide (/=) which follows
    matrix *= 2
    matrix -= 1
    matrix /= np.sqrt(m)

    return matrix


def create_measurement_model(mm_type, patch_size, compression_percent):
    """
    Create measurement model depending on
    :param mm_type: string defining the measurement model type
    :param patch_size: tuple of ints (patch_height, patch_width)
    :param compression_percent: int
    :return: measurement_model: array, shape = (m, n)
    """
    # TODO: check if seed should be in a file rather than hardcoded
    seed = 1234567890

    patch_height, patch_width = patch_size

    n = patch_height * patch_width

    m = round((1 - compression_percent/100) * n)

    if mm_type.lower() == 'gaussian-rip':
        measurement_model = create_gaussian_rip_matrix(size=(m, n), seed=seed)
    elif mm_type.lower() == 'gaussian-orth':
        measurement_model = create_gaussian_orth_matrix(size=(m, n), seed=seed)
    elif mm_type.lower() == 'bernoulli-rip':
        measurement_model = create_bernoulli_rip_matrix(size=(m, n), seed=seed)
    elif mm_type.lower() == 'identity':
        if m is not n:
            TypeError('\'Identity\' measurement model only possible without compression')
        measurement_model = np.eye(n)
    else:
        raise NameError('Undefined measurement model type')

    return measurement_model


def plot_image_set(image_list, name_list, fig=None, sub_plt_n_w=4, axis=False, interpolation=None):
    """
    Plot an image set given as a list
    :param image_list: list of images
    :param name_list: list of names
    :param fig: figure obj
    :param sub_plt_n_w: int, number of subplot spaning the width
    :return:
    """
    # TODO: align images 'top'
    if fig is None:
        fig = plt.figure()

    sub_plt_n_h = int(np.ceil(len(image_list) / sub_plt_n_w))
    ax = []
    for i, (im, im_name) in enumerate(zip(image_list, name_list)):
        ax.append(fig.add_subplot(sub_plt_n_h, sub_plt_n_w, i + 1))
        ax[i].imshow(im, cmap='gray', vmin=im.min(), vmax=im.max(), interpolation=interpolation)
        if not axis:
            ax[i].set_axis_off()
        ax[i].set_title('{}\n({}, {})'.format(im_name, im.shape[0], im.shape[1]), fontsize=10)


def plot_image_with_cbar(image, title=None, cmap='gray', vmin=None, vmax=None, ax=None):
    """
    Plot an image with it's colorbar
    :param image: array, shape = (image_height, image_width)
    :param title: option title
    :param cmap: optional cmap
    :param vmin: optional vmin
    :param vmax: optional vmax
    :param ax: optional axis
    :return:
    """
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title is not None:
        ax.set_title('{}'.format(title), fontsize=12)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

























































def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
  """Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  #if img1.ndim != 4:
  #  raise RuntimeError('Input images must have four dimensions, not %d',
  #                     img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  height, width = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    #window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    window = _FSpecialGauss(size, sigma)
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = np.mean(v1 / v2)
  return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
  """Return the MS-SSIM score between `img1` and `img2`.
  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
  Returns:
    MS-SSIM score between `img1` and `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  #if img1.ndim != 4:
  #  raise RuntimeError('Input images must have four dimensions, not %d',
  #                     img1.ndim)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
  levels = weights.size
  downsample_filter = np.ones((2, 2)) / 4.0
  im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
  mssim = np.array([])
  mcs = np.array([])
  for _ in range(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)
    filtered = [convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]]
    im1, im2 = [x[::2, ::2,] for x in filtered]
  return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
          (mssim[levels-1] ** weights[levels-1]))