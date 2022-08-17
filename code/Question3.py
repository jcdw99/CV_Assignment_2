import numpy as np
from PIL import Image

img = Image.open("resources/bricks.jpg")

tl = (355, 505)
bl = (55, 718)
br = (773, 805)
tr = (814, 543)
minx = np.min(np.array([tl[0], bl[0], br[0], tr[0]]))
maxx = np.max(np.array([tl[0], bl[0], br[0], tr[0]]))
miny = np.min(np.array([tl[1], bl[1], br[1], tr[1]]))
maxy = np.max(np.array([tl[1], bl[1], br[1], tr[1]]))

img.putpixel(tl, (255, 0, 0))
img.putpixel(bl, (255, 0, 0))
img.putpixel(tr, (255, 0, 0))
img.putpixel(br, (255, 0, 0))

cropped = img.crop((minx, miny, maxx, maxy))


cropped.show()

tl = (296, 0)
tr = (755, 36)
bl = (0, 210)
br = (716, 300)

def get_A_matrix():

    x_s = [tl[0], tr[0], bl[0], br[0]]
    y_s = [tl[1], tr[1], bl[1], br[1]]
    #TL TR BR BL
    size_max = 300
    x_primes = [1, size_max, 1, size_max]
    y_primes = [1, 1, size_max, size_max]
    # x_primes = [0 - (1-minx), 0- (1-minx), size_max- (1-minx), size_max- (1-minx)]
    # y_primes = [0- (1-miny), size_max- (1-miny), size_max- (1-miny), 0- (1-miny)]

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


def merge(data):
    img_data = np.array(img)
    # start coords in original image
    start_x = bl[1]
    start_y = bl[0]
    rows_covered = 0
    for i in reversed(range(data.shape[0])):
        for j in range(data.shape[1]):
            img_data[start_x - rows_covered][start_y + j] = data[i][j]
        rows_covered = rows_covered + 1

    img_data.astype(np.uint8)
    true_data = np.array(img)
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            black = [0,0,0]
            if list(img_data[i][j]) ==  black:
                img_data[i][j] = true_data[i][j]

    Image.fromarray(img_data.astype(np.uint8)).show()

mat = get_A_matrix()

U, S, V = np.linalg.svd(mat)
H = (V[8]).reshape((3,3))
trans_poster = applyhomography(cropped, H)
Image.fromarray(trans_poster).show()