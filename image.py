from PIL import Image
import glob
import sys


image_path=sys.argv[1]


def modify_and_save_image(filename, transposeType):
    im = Image.open(filename)
    out = im.transpose(transposeType)
    out.save(filename + str(transposeType) + '.png')

for filename in glob.glob(image_path + '/*.png'): #assuming gif
    modify_and_save_image(filename, Image.FLIP_LEFT_RIGHT)
    modify_and_save_image(filename, Image.FLIP_TOP_BOTTOM)
    modify_and_save_image(filename, Image.ROTATE_90)
    modify_and_save_image(filename, Image.ROTATE_180)
    modify_and_save_image(filename, Image.ROTATE_270)
