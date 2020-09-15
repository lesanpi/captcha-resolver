import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio actual.
cwd = os.getcwd()

SOURCE_PATH = 'source'  # 'data/source/jpg'
TARGET_PATH = 'target'  # 'data/target/jpg'

# SIZE (200, 50)
IMG_HEIGHT = 50
IMG_WIDTH = 200

# Imagenes en la carpeta
imagesJPG = [f for r, d, f in os.walk(SOURCE_PATH)]


def resize(inimg, tgimg, height, width):

    # inimg = tf.image.resize(
    #    inimg, [int(height), int(width)])
    # tgimg = tf.image.resize(
    #    tgimg, [int(height), int(width)])

    return inimg, tgimg


def normalizeImg(inimg, tgimg):
    inimg = (inimg / 127.5) - 1
    tgimg = (tgimg / 127.5) - 1

    return inimg, tgimg


def random_filter(inimg, tgimg):
    # Hacemos mas grandes para luego hacer un crop random
    inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT*1.1, IMG_WIDTH*1.1)

    # Juntamos las imagenes para que el corte se haga en conjunto
    stack_images = tf.stack([inimg, tgimg], axis=1)

    # Crop
    cropped_images = tf.image.random_crop(
        stack_images, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    inimg, tgimg = cropped_images[0], cropped_images[1]

    return inimg, tgimg


def load_image(filename, augmented=True):

    # Decodificamos las imagenes jpg y solo obtenemos los canales RGB
    inimg = tf.cast(tf.image.decode_jpeg(
        tf.io.read_file(SOURCE_PATH + '/' + filename)), tf.float32)[..., :3]
    tgimg = tf.cast(tf.image.decode_jpeg(
        tf.io.read_file(TARGET_PATH + '/' + filename)), tf.float32)[..., :3]

    # Ajustamos el tamano al deseado
    inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)

    # Aumentacion de la data
    # if augmented:
    #    inimg, tgimg = random_filter(inimg, tgimg)

    # Normalizamos los valores de [0, 255] a [-1, 1]
    inimg, tgimg = normalizeImg(inimg, tgimg)

    return inimg, tgimg


@tf.function()
def load_train_image(filename):
    return load_image(filename, True)


@tf.function()
def load_test_image(filename):
    return load_image(filename, False)


# Urls y separando train y test.
images = imagesJPG[0]
n_images = len(images)
porcent = int(n_images*0.8)
train_images = images[:porcent]
test_images = images[porcent:]

# plt.imshow((load_train_image(imagesJPG[0][0])[0] + 1) / 2)
# plt.show()

# Datasets
# Train
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.map(
    load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = train_dataset.batch(1)

# Test
test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
test_dataset = test_dataset.map(
    load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.batch(1)

# Capas de downsample del Encoder.


def downsample(filters, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # Capa convolucional
    result.add(
        tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=not apply_batchnorm))

    # Batch normalization
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    # LeakyRelu
    result.add(tf.keras.layers.LeakyReLU())

    return result

# Capa de upsampling del decoder.


def upsample(filters, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # Convolucional inversa
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, 4, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    # BN
    result.add(tf.keras.layers.BatchNormalization())

    # Aplicamos Dropout o no
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    # Capa de activacion ReLU
    result.add(tf.keras.layers.ReLU())

    return result
