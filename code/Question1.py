
import numpy as np
from PIL import Image

'''
    Uses bilinear interpolation to transform an input image A according to a
    given 3-by-3 projective transformation matrix H.
    
    Notes:
    
    1. This function follows the (x,y) convention for pixel coordinates,
        which differs from the (row,column) convention. The matrix H must be
        set up accordingly.
    
    2. The size of the output is determined automatically, and the output is
        determined automatically, and the output will contain the entire
        transformed image on a white background. This means that the origin of
        the output image may no longer coincide with the top-left pixel. In
        fact, after executing this function, the true origin (0,0) will be
        located at point (1-minx, 1-miny) in the output image (why?).
    
'''
    

"""
    Applies the transformation specified by the provided matrix, to an input image. The Input Image may be a PILLOW Image,
    or a NumpyArray.
"""
def apply_transform(img, matrix):
    # convert potential Image Object to ndArray
    img_data = np.array(img).astype(float)
    # get rows, cols, channels of image data
    rows, cols, width = img_data.shape
    # forward transform the corners, 
    top_left = np.dot(matrix, np.array([0, 0, 1]))
    bottom_left = np.dot(matrix, np.array([rows-1, 0, 1]))
    top_right = np.dot(matrix, np.array([0, cols-1, 1]))
    bottom_right = np.dot(matrix, np.array([rows-1, cols-1, 0]))

    # get the bounding box surrounding the forward-mapped image, which is subsequently mapped to (0,0) during array initialization, from (minx, miny)
    min_x = np.floor(np.min([top_left[0], bottom_left[0], top_right[0], bottom_right[0]]))
    max_x = np.ceil(np.max([top_left[0], bottom_left[0], top_right[0], bottom_right[0]]))
    min_y = np.floor(np.min([top_left[1], bottom_left[1], top_right[1], bottom_right[1]]))
    max_y = np.ceil(np.max([top_left[1], bottom_left[1], top_right[1], bottom_right[1]]))

    x_range = int(max_x - min_x)
    y_range =int(max_y - min_y)

    #init output image
    output = np.zeros((y_range, x_range, width))
    
    # similarly to Assignment1, we do inverse mapping, IE -> input_coords = [mat^-1][output_coords]
    # this avoids a mesh-like grid related to floating-point casting, occuring when we do forward mapping
    inv_mat = np.linalg.inv(matrix)

    # perform bilinear interpolation
    def do_bilinear(in_x, in_y, output, x, y):
        xpf = int(np.floor(in_x))
        xpc = xpf + 1
        ypf = int(np.floor(in_y))
        ypc = ypf + 1
        if ((xpf >= 0) and (xpc < rows) and (ypf >= 0) and (ypc < cols)):
            output[y,x,:] =(xpc - in_x)*(ypc - in_y)*img_data[ypf,xpf,:] \
                        + (xpc - in_x)*(in_y - ypf)*img_data[ypc,xpf,:] \
                        + (in_x - xpf)*(ypc - in_y)*img_data[ypf,xpc,:] \
                        +  (in_x - xpf)*(in_y - ypf)*img_data[ypc,xpc,:]
             
    for x in range(x_range):
        for y in range(y_range):
            # compensate for the shift in B's origin
            point = np.array([x + min_x, y + min_y, 1])
            # inverse mapping, calculates the corresponding point in the original image
            corr = np.dot(inv_mat, point)  
            # do bilinear transformation
            do_bilinear(corr[0], corr[1], output, x, y)

    pic = Image.fromarray(output.astype(np.uint8))
    pic.show()

def get_rotation_matrix(theta):
    theta = np.radians(theta)
    rotation_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [      0,              0,       1]]
    )
    return rotation_mat

# apply_transform(Image.open('resources/bricks.jpg'), get_rotation_matrix(40))


# Image.fromarray(applyhomography(Image.open("resources/bricks.jpg"), get_rotation_matrix(30))).show()