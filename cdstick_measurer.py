import cv2
from PIL import Image, ImageDraw
from numpy import ndarray
import glob
import numpy as np
import copy


def find_single_edge(data:ndarray):
    h, w = data.shape[0], data.shape[1]
    win_size = 5
    half_w = win_size // 2

    # init to be maximum
    accu_hist = np.zeros(w)
    for c in range(half_w, w-half_w):
        sub_data = data[:, c-half_w:c+half_w+1]
        v = np.sum(sub_data, axis=0)
        accu_hist[c] = np.sum(v)

    # 计算梯度
    grad = np.zeros_like(accu_hist)
    for c in range(half_w+1, w-half_w):
        grad[c] = accu_hist[c-1] - accu_hist[c]

    ind1 = np.argmax(grad[half_w+1:w-half_w-1])
    ind2 = np.argmin(grad[half_w+1:w-half_w-1])-1

    return ind1+half_w+1, ind2+half_w+1


def find_both_edge(data:ndarray):
    h, w = data.shape[0], data.shape[1]

    scale = 5
    lhalf_img = data[h//scale : (scale-1)*h//scale, 0:w//2]
    l_ind1, l_ind2 = find_single_edge(lhalf_img)
    rhalf_img = data[h//scale : (scale-1)*h//scale, w//2:w]
    r_ind1, r_ind2 = find_single_edge(rhalf_img)
    r_ind1 = r_ind1 + w//2
    r_ind2 = r_ind2 + w // 2

    return l_ind1, l_ind2, r_ind1, r_ind2

#
def get_stick_size(img_path):
    # gray model
    img = Image.open(img_path).convert('L')

    # find vertical edges
    data = np.array(img)
    l_ind1, l_ind2, r_ind1, r_ind2 = find_both_edge(data)

    # need correction
    if abs(r_ind2-r_ind1) > abs(l_ind2 - l_ind1):
        d = abs(r_ind2 - r_ind1) - abs(l_ind2-l_ind1)
        r_ind1 = r_ind1 + d // 2
        r_ind2 = r_ind2 - d // 2

    # # rotate image and find horizontal edges
    # rimg = img.rotate(90, expand=1)
    # data = np.array(rimg)
    # t_ind, b_ind = find_both_edge(data)

    return l_ind1, l_ind2, r_ind1, r_ind2


def show_result(l1,l2,r1,r2,img_path):
    img = Image.open(img_path)
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)
    # 左边的线
    sp, ep = (l1,0), (l1, h-1)
    draw.line([sp, ep], fill="red", width=2)
    sp, ep = (l2,0), (l2, h-1)
    draw.line([sp, ep], fill="blue", width=2)
    # 右边的线
    sp, ep = (r1, 0), (r1, h-1)
    draw.line([sp, ep], fill="blue", width=2)
    sp, ep = (r2,0), (r2, h-1)
    draw.line([sp, ep], fill="red", width=2)

    img.show()


def test_cdstick():
    image_path = "testimg/cdstick"
    img = image_path + "/1.jpg"
    # 返回inner size和outer size
    l1,l2,r1,r2 = get_stick_size(img)
    print(l1, l2, r1, r2)
    show_result(l1, l2, r1, r2, img)


if __name__ == '__main__':
    test_cdstick()

