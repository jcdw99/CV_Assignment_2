
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    return pic

def applyhomography(A,H):
    # cast the input image to double precision floats
    A = np.array(A).astype(float)
    
    # determine number of rows, columns and channels of A
    m, n, c = A.shape
    
    # determine size of output image by forward−transforming the four corners of A
    p1 = np.dot(H,np.array([0,0,1]).reshape((3,1))); p1 = p1/p1[2];
    p2 = np.dot(H,np.array([n-1, 0,1]).reshape((3,1))); p2 = p2/p2[2];
    p3 = np.dot(H,np.array([0, m-1,1]).reshape((3,1))); p3 = p3/p3[2];
    p4 = np.dot(H,np.array([n-1,m-1,1]).reshape((3,1))); p4 = p4/p4[2];
    minx = np.floor(np.amin([p1[0], p2[0], p3[0] ,p4[0]]));
    maxx = np.ceil(np.amax([p1[0], p2[0], p3[0] ,p4[0]]));
    miny = np.floor(np.amin([p1[1], p2[1], p3[1] ,p4[1]]));
    maxy = np.ceil(np.amax([p1[1], p2[1], p3[1] ,p4[1]]));
    nn = int(maxx - minx)
    mm = int(maxy - miny)

    print(nn, ' by ', mm)

    # initialise output with white pixels
    B = np.zeros((mm,nn,c))

    # pre-compute the inverse of H (we'll be applying that to the pixels in B)
    Hi = np.linalg.inv(H)
    
    # Loop  through B's pixels
    for x in range(nn):
        for y in range(mm):
            # compensate for the shift in B's origin
            p = np.array([x + minx, y + miny, 1]).reshape((3,1))
            
            # apply the inverse of H
            pp = np.dot(Hi,p)

            # de-homogenise
            xp = pp[0]/pp[2]
            yp = pp[1]/pp[2]
        
            # perform bilinear interpolation
            xpf = int(np.floor(xp)); xpc = xpf + 1;
            ypf = int(np.floor(yp)); ypc = ypf + 1;


            if ((xpf >= 0) and (xpc < n) and (ypf >= 0) and (ypc < m)):
                B[y, x,:] = (xpc - xp)*(ypc - yp)*A[ypf,xpf,:] \
                            + (xpc - xp)*(yp - ypf)*A[ypc,xpf,:] \
                            + (xp - xpf)*(ypc - yp)*A[ypf,xpc,:] \
                            +  (xp - xpf)*(yp - ypf)*A[ypc,xpc,:] \


    return Image.fromarray(B.astype(np.uint8))

def get_rotation_matrix(theta):
    theta = np.radians(theta)
    rotation_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [      0,              0,       1]]
    )
    return rotation_mat

def get_scale_matrix(scale):
    mat = np.array( 
        [[scale, 0, 0],
        [0, scale,  0],
        [ 0,  0,   1]]
    )
    return mat 

def get_sheerX_matrix(scale):
    mat = np.array( 
        [[1, scale, 0],
        [0, 1,  0],
        [ 0,  0,   1]]
    )
    return mat 

def get_sheerY_matrix(scale):
    mat = np.array( 
        [[1, 0, 0],
        [scale, 1,  0],
        [ 0,  0,   1]]
    )
    return mat 



result = applyhomography(Image.open('resources/bricks.jpg'), get_rotation_matrix(70))
plt.imshow(result)
plt.savefig('output/rotate70.jpg')
plt.clf()
exit()

result = applyhomography(Image.open('resources/bricks.jpg'), get_scale_matrix(0.5))
plt.imshow(result)
plt.savefig('output/puppies0.5.jpg')
plt.clf()


# Image.fromarray(applyhomography(Image.open("resources/bricks.jpg"), get_rotation_matrix(30))).show()