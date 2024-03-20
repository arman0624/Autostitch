This program combines a series of horizontally overlapping photographs into a single panoramic image using an ORB feature detector and descriptor. ORB is used to first detect discriminating features in the images and find the best matching features in the other images. Then, using RANSAC, the photographs are automatically aligned and then blended,resulting in a single seamless panorama. There is a graphical interface that lets you view the results of your own creations or wioth some provided test images.

The program consists of a pipeline of tabs visualized through AutostitchUI that will operate on images or intermediate results to produce the final panorama output.  

The steps required to create a panorama are listed below. There are two ways to stitch a panorama: using translations and homographies, where the input images are alligned directly.


Step 1 - Take pictures on a tripod (or handheld)

Step 2 - Extract features

Step 3 - Match features

Step 4 - Align neighboring pairs using RANSAC

Step 5 - Write out list of neighboring translations

Step 6 - Correct for drift

Step 7 - Read in [warped] images and blend them

Step 8 - Crop the result and import into a viewer

To run the gui, use this command: python gui.py 