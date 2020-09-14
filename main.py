import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio actual.
cwd = os.getcwd()

SOURCE_PATH = 'data/source/jpg'
TARGET_PATH = 'data/target/jpg'

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


plt.imshow((load_train_image(imagesJPG[0][0])[0] + 1) / 2)
plt.show()
