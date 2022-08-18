import numpy as np
from PIL import Image

kayak1 = Image.open('resources/kayak1.jpg')
kayak2 = Image.open('resources/kayak2.jpg')
                # Pointy     # waterfall   #tree   #crack
kayak1_points = [(358, 211), (466,142), (384,52), (418,271)]
kayak2_points = [(135, 259), (235, 181), (155,98), (200,307)]

src_tl = kayak1_points[0]
src_tr = kayak1_points[1]
src_bl = kayak1_points[2]
src_br = kayak1_points[3]

dest_tl = kayak2_points[0]
dest_tr = kayak2_points[1]
dest_bl = kayak2_points[2]
dest_br = kayak2_points[3]

def get_A_matrix():

    x_s = [src_tl[0], src_tr[0], src_bl[0], src_br[0]]
    y_s = [src_tl[1], src_tr[1], src_bl[1], src_br[1]]
    #TL TR BR BL
    x_primes = [dest_tl[0], dest_tr[0], dest_bl[0], dest_br[0]]
    y_primes = [dest_tl[1], dest_tr[1], dest_bl[1], dest_br[1]]

    col_1 = [x_s[i//2] if i % 2 == 0 else 0 for i in range(2*len(x_s))]
    col_2 = [y_s[i//2] if i % 2 == 0 else 0 for i in range(2*len(y_s))]
    col_3 = [1 if i % 2 == 0 else 0 for i in range(2*len(y_s))]
    col_4 = [x_s[i//2] if i % 2 != 0 else 0 for i in range(2*len(x_s))]
    col_5 = [y_s[i//2] if i % 2 != 0 else 0 for i in range(2*len(y_s))]
    col_6 = [1-i for i in col_3]
    # col_7 first term is -x1, -y1, -x2, -y2...-x4, -y4 and x1, x1, x2, x2...x4, x4
    col_7 = np.array([x_s[i//2] for i in range(2*len(x_s))]) * np.array([-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(y_primes))])
    # col_8 first term is -x1, -y1, -x2, -y2...-x4, -y4 and y1, y1, y2, y2...y4, y4
    col_8 = np.array([y_s[i//2] for i in range(2*len(y_s))]) * np.array([-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(y_primes))])
    # col_9 simply alternates between negative entries in x_s and y_s
    col_9 = [-x_primes[i//2] if i % 2 == 0 else -y_primes[i//2] for i in range(2*len(x_primes))]
    matrix = np.array([col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9]).transpose()

    return matrix

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


    return B.astype(np.uint8)

mat = get_A_matrix()
U, S, V = np.linalg.svd(mat)
H = (V[8]).reshape((3,3))
stretched = applyhomography(kayak1, H)
#             Pointy      Waterfall     Tree       Crack
padded = [(964, 734), (1065, 655), (985, 571), (1028, 780)]
show_mode = False

def stitch():
    stretched_data = np.pad(stretched, [(400, 0), (0, 400), (0,0)], mode='constant', constant_values=0)
    stretched_pic = Image.fromarray(stretched_data)

    if show_mode:
        for point in padded:
            stretched_pic.putpixel(point, (255,0,0))
            stretched_pic.putpixel((point[0]-1, point[1]-1), (255,0,0))
            stretched_pic.putpixel((point[0]-1, point[1]+1), (255,0,0))
            stretched_pic.putpixel((point[0]+1, point[1]-1), (255,0,0))
            stretched_pic.putpixel((point[0]+1, point[1]+1), (255,0,0))
        stretched_pic.show()
    kayak2_cropped = kayak2.crop((kayak2_points[2][0], 0, kayak2.size[0], kayak2.size[1]))
    pasted = stretched_pic.copy()
    Image.Image.paste(pasted, kayak2_cropped, (padded[2][0], padded[2][1] - 100))
    # pasted.save('output/pasted1.jpg')
    pasted.show()
    exit()
    # for every row of the kayak2 pic
    for i in range(kayak2_cropped.size[0]):
        # for every col of the kayak2 pic
        for j in range(kayak2_cropped.size[1]):
            # if a pixel exists in this location in the stretched padded image
            target = (padded[2][0] + i, padded[2][1] - 100 + j)
            if stretched_pic.getpixel(target) != (0,0,0):
                pasted.putpixel(target, stretched_pic.getpixel(target)) 
            

    pasted.save('output/pasted2.jpg')
    exit()

stitch()
