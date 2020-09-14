from PIL import Image, ImageDraw, ImageFont
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_PATH = 'samples/'
TARGET_PATH = 'data/target/jpg/'

# Directorio actual.
cwd = os.getcwd()

sample_path = os.path.join(cwd, SAMPLE_PATH)
target_path = os.path.join(cwd, TARGET_PATH)

# Font Family y Font Size.
font = ImageFont.truetype('fonts/Roboto-Bold.ttf', 36)


def generateImage(text):
    # Creando lienzo, rgb, tamanio y fondo.
    img = Image.new('RGB', (200, 50), color='white')

    d = ImageDraw.Draw(img)
    # Escribimos sobre la imagen
    d.text((50, 5), text, font=font, fill=(0, 0, 0))

    # plt.imshow(img)
    # plt.show()
    img.save(os.path.join(TARGET_PATH, str(text) + ".jpg"))


def generateTargetImages():
    # Clear output directory
    shutil.rmtree(TARGET_PATH, ignore_errors=True)
    os.mkdir(TARGET_PATH)

    # r=root, d=directories, f = files
    for r, d, f in os.walk(SAMPLE_PATH):
        for file in f:
            if '.jpg' in file:
                filename, file_extension = os.path.splitext(file)
                generateImage(filename)


generateTargetImages()
