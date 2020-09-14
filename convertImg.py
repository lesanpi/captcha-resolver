import os
import shutil

from PIL import Image

source = 'samples/'
target = "source/"

cwd = os.getcwd()

pathsource = os.path.join(cwd, source)
pathtarget = os.path.join(cwd, target)


def convertImage(r, filename, file):

    img = Image.open(os.path.join(r, file)).convert('RGB')
    img.save(os.path.join(target, str(filename) + ".jpg"))


def convertRGB():
    shutil.rmtree(pathtarget, ignore_errors=True)
    os.mkdir(pathtarget)

    # r=root, d=directories, f = files
    for r, d, f in os.walk(pathsource):
        for file in f:
            if '.jpg' in file:
                filename, file_extension = os.path.splitext(file)
                convertImage(r, filename, file)
            elif '.png' in file:
                filename, file_extension = os.path.splitext(file)
                convertImage(r, filename, file)


convertRGB()
