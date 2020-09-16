from IPython import display
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio actual.
cwd = os.getcwd()

SOURCE_PATH = 'source'  # 'data/source/jpg'
TARGET_PATH = 'target'  # 'data/target/jpg'

# SIZE (200, 50)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Imagenes en la carpeta
imagesJPG = [f for r, d, f in os.walk(SOURCE_PATH)]


@tf.function()
def resize(inimg, tgimg, height, width):

    inimg = tf.image.resize(
        inimg, [int(height), int(width)])
    tgimg = tf.image.resize(
        tgimg, [int(height), int(width)])

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


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128),  # (bs, 64, 64, 128)
        downsample(256),  # (bs, 32, 32, 256)
        downsample(512),  # (bs, 16, 16, 512)
        downsample(512),  # (bs, 8, 8, 512)
        downsample(512),  # (bs, 4, 4, 512)
        downsample(512),  # (bs, 2, 2, 512)
        downsample(512),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512),  # (bs, 16, 16, 1024)
        upsample(256),  # (bs, 32, 32, 512)
        upsample(128),  # (bs, 64, 64, 256)
        upsample(64),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling a traves del modelo
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling y uniendo las skips connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# Ejemplo
inp, re = load_train_image(train_images[0])
inp = (inp + 1) * 127.5
re = (re + 1) * 127.5
generator = Generator()
# gen_out = generator(inp[tf.newaxis, ...], training=False)

# plt.imshow(re / 255)
# plt.show()


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    # Input y Target
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256)(down2)  # (bs, 32, 32, 256)
    down4 = downsample(256)(down3)

    # zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    # conv = tf.keras.layers.Conv2D(512, 4, strides=1,
    #                              kernel_initializer=initializer,
    #                              use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    #batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    #leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    # zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    # last = tf.keras.layers.Conv2D(1, 4, strides=1,
    #                              kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(down4)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# Optimizadores
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        # if (epoch + 1) % 20 == 0:
        #  checkpoint.save(file_prefix = checkpoint_prefix)

        # print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
        #                                                    time.time()-start))


EPOCHS = 150
fit(train_dataset, EPOCHS, test_dataset)
