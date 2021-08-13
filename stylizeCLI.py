import argparse

parser = argparse.ArgumentParser(description='Apply stylization to images using model by Ghiasi et al.')
parser.add_argument('styleImagesFolder', type=str,
                   help = 'path to the folder containing style images')
parser.add_argument('contentImagesFolder', type=str,
                   help = 'path to the folder containing content images')
parser.add_argument('outputFolder', type=str,
                   help = 'path to save the stylized images at ')
parser.add_argument('--convertWebp', action='store_true',
                   help= 'convert images in webp format to jpeg')
parser.add_argument('--skipCrop', action='store_true',
                   help= 'prevents cropping images to 1024x1024 when either dimension is greater than 1920')
args = parser.parse_args()

import os

if not os.path.isdir(args.styleImagesFolder):
    print("Could not find folder at passed style image folder path")
    exit()

if not os.path.isdir(args.contentImagesFolder):
    print("Could not find folder at passed content image folder path")
    exit()

if not os.path.isdir(args.outputFolder):
    print("Could not find folder at passed output folder path")
    exit()


import functools
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

#version of load_noadj that loads from filepath (for use with content images)
@functools.lru_cache(maxsize=None)
def load_fileimage_noadj(image_filepath, preserve_aspect_ratio=True, shrinkIf2Big=True):
    try:
        img = tf.io.decode_image(tf.io.read_file(image_filepath), channels=3, dtype=tf.float32)[tf.newaxis, ...]
    except tf.errors.InvalidArgumentError:
        #print("Image at", image_filepath, "might've been webp, resaving as jpg")
        Image.open(image_filepath).convert('RGB').save(image_filepath, 'jpeg')
        img = tf.io.decode_image(tf.io.read_file(image_filepath), channels=3, dtype=tf.float32)[tf.newaxis, ...]

    if shrinkIf2Big and ((img.shape[2] > 1920) or (img.shape[3] > 1920) or (img.shape[1] > 1920)):
        img = tf.image.resize(img, (1024, 1024), preserve_aspect_ratio=True)
    return img

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

#version of load function that loads from filepath (for use with style images)
@functools.lru_cache(maxsize=None)
def load_fileimage(image_filepath, image_size=(256, 256), preserve_aspect_ratio=True):
    try:
        img = tf.io.decode_image(tf.io.read_file(image_filepath), channels=3, dtype=tf.float32)[tf.newaxis, ...]
    except tf.errors.InvalidArgumentError:
        #print("Image at", image_filepath, "might've been webp, resaving as jpg")
        Image.open(image_filepath).convert('RGB').save(image_filepath, 'jpeg')
        img = tf.io.decode_image(tf.io.read_file(image_filepath), channels=3, dtype=tf.float32)[tf.newaxis, ...]

    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def returnStyleImagesFromFilepaths(style_filepaths):
    style_images = {k: load_fileimage(v, (256, 256)) for k, v in style_filepaths.items()}
    style_images = {k: tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME') for k, style_image in style_images.items()}
    return style_images

#use the filenames (exluding the last four characters, e.g. .png), as the keys in the dict
style_filepaths = {i[:-4]:args.styleImagesFolder+"/"+i for i in os.listdir(args.styleImagesFolder)}
style_images = returnStyleImagesFromFilepaths(style_filepaths)

content_filepaths = {i[:-4]:args.contentImagesFolder+"/"+i for i in os.listdir(args.contentImagesFolder)}
content_images = {k: load_fileimage_noadj(v, shrinkIf2Big=True) for k, v in content_filepaths.items()}

print("Found", len(style_images), "style images and", len(content_images), "content images.")

print("Loading stylization network...")
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("Network loaded!")

print("Creating", len(style_images) * len(content_images), "stylized images...")

my_dpi = 120
def applyAllStylesToAllWhileSaving(content_links, style_links):
    for content_name in content_links:
        for style_name in style_links:
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                        tf.constant(style_images[style_name]))[0]

            height = stylized_image[0].shape[0] / 1.5
            width = stylized_image[0].shape[1] / 1.5
            plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
            plt.imshow(stylized_image[0])
            plt.axis('off')
            plt.savefig(args.outputFolder+'/'+style_name+'-'+content_name+'.png', bbox_inches='tight', pad_inches=0)
            plt.close()

applyAllStylesToAllWhileSaving(content_filepaths, style_filepaths)

print("Done!")
