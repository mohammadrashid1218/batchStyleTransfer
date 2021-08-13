## Quickly stylize images in batches 

This is a python command line app for applying [Ghiasi et al.'s fast style transfer network](https://arxiv.org/abs/1705.06830)
to many images at once without needing to install the magenta environment. Includes convenience features like webp conversion and autocropping based on image size. 

Requirements: 

```
tensorflow=>2.5.0
tensorflow-hub>=0.12.0
matplotlib>=3.1.2
```
If you want images in webp format to be automaticaly converted to jpeg, `Pillow>=7.2.0` is also required. 

Example usage: 
`python stylizeCLI.py style_im_folder content_im_folder output_folder --convertWebp`     

Full syntax: 

```
usage: stylizeCLI.py [-h] [--convertWebp] [--skipCrop]
                     styleImagesFolder contentImagesFolder outputFolder


positional arguments:  
  styleImagesFolder    path to the folder containing style images  
  contentImagesFolder  path to the folder containing content images  
  outputFolder         path to save the stylized images at  

optional arguments:  
  -h, --help           show this help message and exit  
  --convertWebp        convert images in webp format to jpeg  
  --skipCrop           prevents cropping images to 1024x1024 when either  
                       dimension is greater than 1920  
```


Every style in the style images folder will be applied to every image in the content images folder.

If the script is working correctly, you should see the below messages:
```
Found 2 style images and 1 content images.
Loading stylization network...
Network loaded!
Creating 2 stylized images
```
alongside a bunch of tensorflow related messages.

The model won't work on images in webp format, so it's recommended to pass `--convertWebp` to convert any webp images to jpeg (though the file ending will not be changed). `Pillow>=7.2.0` is a requirement for this conversion. 

Any content image with either dimension greater than 1920 pixels will be cropped to 1024 x 1024. 
This is recommended because processing takes very long on large images. 
Pass `--skipCrop` to prevent this. Note that style images are always cropped to 256 x 256 because the network doesn't work as well with other style resolutions. 

### Solution for OSError after model download interruption

If you get the message `OSError: SavedModel file does not exist at _dirpath_`, you need to
delete the directory mentioned in the error. This error occurs if the model download didn't 
finish properly, which can happen if you Ctrl+C out of the script within the first 
few seconds. 