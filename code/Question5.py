import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from random import sample



# orig_pics = [Image.open('resources/fifaimages/' + str(i) + '.jpg') for i in range(1, 13)]
# template = Image.open('resources/fifaimages/template.jpg')
# pasted = [i.copy() for i in orig_pics]
# for i in pasted:
#     Image.Image.paste(i, template, (0,0))
RANSAC_THRESH = 10
ITERS = 3000

def gen_pasted_pic(i):
    orig = Image.open('resources/fifaimages/' + str(i) + '.jpg')
    template = Image.open('resources/fifaimages/template.jpg')
    Image.Image.paste(orig, template, (0,0))
    return orig
def file_to_data(file_num):
    path = 'resources/fifaimages/fifasiftmatches/siftmatches_' + str(file_num) + '.txt'
    df = pd.read_csv(path, sep=',', header=None)
    return df

def draw_mapping(i=None):
    # get list of mapping files to plot
    targets = [i] if i is not None else list(range(1,13))
    output = []
    for target in targets:
        df = file_to_data(target)
        pasted_pic = gen_pasted_pic(target)
        draw = ImageDraw.Draw(pasted_pic)
        x = np.array(pd.to_numeric(df[0]))
        y = np.array(pd.to_numeric(df[1]))
        xprime = np.array(pd.to_numeric(df[2]))
        yprime = np.array(pd.to_numeric(df[3]))
        
        for i in range(len(x)):
            draw.ellipse(((x[i]-1, y[i]-1, x[i]+1, y[i]+1)), fill="yellow")
            draw.ellipse(((xprime[i]-1, yprime[i]-1, xprime[i]+1, yprime[i]+1)), fill="yellow")
            draw.line([(x[i], y[i]), (xprime[i], yprime[i])], fill='yellow', width=1)
        output.append(pasted_pic)
        pasted_pic.show()
        pasted_pic.save("output/fifa" + str(target) + '_map.jpg')
    return output

def get_A_matrix(df):

    x = np.array(df[0])
    y = np.array(df[1])
    xprime = np.array(df[2])
    yprime = np.array(df[3])

    x_s = [x[i] for i in range(len(x))]
    y_s = [y[i] for i in range(len(y))]
    #TL TR BR BL
    x_primes = [xprime[i] for i in range(len(xprime))]
    y_primes = [yprime[i] for i in range(len(yprime))]

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

def do_RANSAC(target):
    data = file_to_data(target)
    best_inlier_set = []
    for k in range(ITERS):
        index_options = list(range(len(data)))
        mask = np.array([False] * len(data))
        sample_index = sample(index_options, 4)
        for i in sample_index:
            mask[i] = True
        sample_data = data.loc[mask]
        A = get_A_matrix(sample_data)
        U, S, V = np.linalg.svd(A)
        H = (V[8]).reshape((3,3))
        inlier_set = []
        for data_dex in range(len(data)):
            data_point = np.array(data.loc[data_dex])
            dest = np.dot(H, np.array([data_point[0], data_point[1], 1]))
            dest = (dest / dest[2])[:-1]
            diff = (np.linalg.norm(dest - np.array([data_point[2], data_point[3]])))
            if diff < RANSAC_THRESH:
                inlier_set.append(data_point)
        if len(inlier_set) > len(best_inlier_set):
            best_inlier_set = inlier_set
    print(np.array(best_inlier_set))
    return np.array(best_inlier_set)


def draw_ransac_mappings(target):
    data_set = do_RANSAC(target)
    pasted_pic = gen_pasted_pic(target)
    draw = ImageDraw.Draw(pasted_pic)
    
    for i in range(len(data_set)):
        draw.ellipse(((data_set[i][0]-1, data_set[i][1]-1, data_set[i][0]+1, data_set[i][1]+1)), fill="yellow")
        draw.ellipse(((data_set[i][2]-1, data_set[i][3]-1, data_set[i][2]+1, data_set[i][3]+1)), fill="yellow")
        draw.line([(data_set[i][0], data_set[i][1]), (data_set[i][2], data_set[i][3])], fill='yellow', width=1)

    pasted_pic.show()
    pasted_pic.save("output/fifa_" + str(target) + '_sac.jpg')

def compare_orig_ransac(i):
    draw_mapping(i)
    draw_ransac_mappings(i)

def recompute_H_ALL(i):
    points = do_RANSAC(i)
    points = pd.DataFrame(points)
    A = get_A_matrix(points)
    U, S, V = np.linalg.svd(A)
    H = (V[8]).reshape((3,3))

    temp = Image.open('resources/fifaimages/template.jpg')
    pasted = gen_pasted_pic(i)
    dims = temp.size
    # now forward map the 4 corners
    new_00 = np.dot(H, np.array([0, 0, 1]))
    new_00 = (new_00 / new_00[2])[:-1]
    new_00 = tuple(new_00.astype(int))

    new_10 = np.dot(H, np.array([dims[0], 0, 1]))
    new_10 = (new_10 / new_10[2])[:-1]
    new_10 = tuple(new_10.astype(int))

    new_01 = np.dot(H, np.array([0, dims[1], 1]))
    new_01 = (new_01 / new_01[2])[:-1]
    new_01 = tuple(new_01.astype(int))

    new_11 = np.dot(H, np.array([dims[0], dims[1], 1]))
    new_11 = (new_11 / new_11[2])[:-1]
    new_11 = tuple(new_11.astype(int))
    # print(pasted.size)
    # print(new_00)
    # print(new_01)
    # print(new_10)
    # print(new_11)
    # exit()
    draw_green_borders(pasted, (0,0), (dims[0], 0), (0, dims[1]), (dims[0], dims[1]), new_00, new_10, new_01, new_11)
    pasted.save("output/fifa" + str(i) + '.jpg')
    pasted.show()


def draw_green_borders(pic, point1, point2, point3, point4, point5, point6, point7, point8):
    draw = ImageDraw.Draw(pic)
    draw.line([point1, point2], fill='green', width=5)
    draw.line([point2, point4], fill='green', width=5)
    draw.line([point4, point3], fill='green', width=5)
    draw.line([point3, point1], fill='green', width=5)
    draw.line([point5, point6], fill='green', width=5)
    draw.line([point6, point8], fill='green', width=5)
    draw.line([point8, point7], fill='green', width=5)
    draw.line([point7, point5], fill='green', width=5)


do_RANSAC(11)
    