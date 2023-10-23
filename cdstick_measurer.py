import cv2
from PIL import Image, ImageDraw
from numpy import ndarray
import glob
import numpy as np
import os
import config as CFG


def find_single_edge(data:ndarray):
    h, w = data.shape[0], data.shape[1]
    win_size = 5
    half_w = win_size // 2

    # init to be maximum
    accu_hist = np.ones(w) * 1000000
    for c in range(half_w, w-half_w):
        sub_data = data[:, c-half_w:c+half_w+1]
        v = np.sum(sub_data, axis=0)
        accu_hist[c] = np.sum(v)
    ind = np.argmin(accu_hist)

    return ind


def find_both_edge(data:ndarray):
    h, w = data.shape[0], data.shape[1]

    scale = 6
    lhalf_img = data[h//scale : (scale-1)*h//scale, 0:w//2]
    l_ind = find_single_edge(lhalf_img)
    rhalf_img = data[h//scale : (scale-1)*h//scale, w//2:w]
    r_ind = find_single_edge(rhalf_img)
    r_ind = r_ind + w//2

    return l_ind, r_ind

#
def get_stick_size(img_path):
    # gray model
    img = Image.open(img_path).convert('L')

    # find vertical edges
    data = np.array(img)
    l_ind, r_ind = find_both_edge(data)

    # rotate image and find horizontal edges
    rimg = img.rotate(90, expand=1)
    data = np.array(rimg)
    t_ind, b_ind = find_both_edge(data)

    return l_ind, r_ind, t_ind, b_ind


def show_result(l,r,t,b,img_path):
    img = Image.open(img_path)
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)
    sp, ep = (l,0), (l, h-1)
    draw.line([sp, ep], fill="red", width=2)
    sp, ep = (r, 0), (r, h-1)
    draw.line([sp, ep], fill="red", width=2)
    sp, ep = (0, t), (w-1, t)
    draw.line([sp, ep], fill="red", width=2)
    sp, ep = (0, b), (w-1, b)
    draw.line([sp, ep], fill="red", width=2)
    img.show()


def test_cdstick():
    image_path = "testimg/cdstick"
    img = image_path + "/5.jpg"
    l,r,t,b = get_stick_size(img)
    print(l, r, t, b)
    show_result(l, r, t, b, img)
    # images = glob.glob(os.path.join(image_path, '*'))
    # for img in images:
    #     std = get_stick_size(img)
    #     print("file={}, std={}".format(img, std))

    # 单张测试
    # res = compare_img_quality("testimg/quality/x20/2.jpg", "testimg/quality/x20/1.jpg")
    # print(res, int(res))


if __name__ == '__main__':
    test_cdstick()

