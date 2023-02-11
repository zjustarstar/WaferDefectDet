import cv2
import os
import config as CFG


def isDefect_byPatchCompare(templepath, testingImg, threshould):
    compSizeX = 100
    compSizeY = 100
    if templepath.find('Template') <= 0:
        return 1, 0
    tpath = templepath.replace('Template', 'OK')
    tpath = os.path.dirname(tpath)
    paths = os.listdir(tpath)
    if len(paths) <= 0:
        return 1, 0

    tmpIMG = cv2.imread(os.path.join(tpath, paths[0]))

    h, w = tmpIMG.shape[:2]
    h2, w2 = testingImg.shape[:2]
    if (h != h2) or (w != w2):
        return 1, 0

    rowCount = int(h / 100)
    colCount = int(w / 100)
    startPoint = 20
    endPoint = 20

    print(compSizeX, compSizeY, rowCount, colCount)

    rslt = []
    Low95Count = 0
    for k in range(rowCount):
        for j in range(colCount):
            tx = j * compSizeX
            ty = k * compSizeY
            ox = j * compSizeX + startPoint
            oy = k * compSizeY + startPoint

            ow = compSizeX
            oh = compSizeY
            tw = compSizeX + startPoint + endPoint
            th = compSizeY + startPoint + endPoint
            if ox + ow > (w - endPoint):
                ow = w - endPoint - ox
            if oy + oh > (h - endPoint):
                oh = h - endPoint - oy

            if tx + tw > (w - endPoint):
                tw = w - endPoint - tx
            if ty > (h - endPoint):
                th = h - endPoint - ty

            # print(ox, oy, tx, ty, tw, th)
            if (ow == compSizeX and oh == compSizeY and tw > 0 and th > 0):
                tmpCutImg = tmpIMG[ty:ty + th, tx:tx + tw]
                curCutImg = testingImg[oy:oy + oh, ox:ox + ow]

                res = cv2.matchTemplate(tmpCutImg, curCutImg, cv2.TM_CCOEFF_NORMED)
                minV, maxV, minL, maxL = cv2.minMaxLoc(res)
                rslt.append(maxV)
                # print(j, k, maxV, maxL)
                if (maxV < threshould):
                    Low95Count = Low95Count + 1
                    # print(Low95Count, j, k, maxV, maxL)
                    # cv2.imshow("tmpCutImg", tmpCutImg)
                    # cv2.imshow("curCutImg", curCutImg)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                # loc = np.where(res >= Similarity)
                # if np.size(loc) != 0:
                #     info_list = ['True']
                # else:
                #     info_list = ['False']
                #
                # info = {'info_list': [info_list]}

    sum = 0
    avrage = 0
    count = len(rslt)
    if count > 0:
        for k in range(len(rslt)):
            sum = sum + rslt[k]
        avrage = sum / count

    avrage = int(avrage * 100) / 100
    # print(min(rslt), max(rslt), avrage, Low95Count)
    if(min(rslt) < threshould - 0.2):
        return 1, avrage
    else:
        if ((min(rslt) < threshould - 0.1) and (Low95Count > 5)):
            return 1, avrage

    return 0, avrage


def cell_abnormal_det(img_path,):
    msg = "OK"

    image = cv2.imread(img_path)
    # 最后两个返回的参数是：是否是缺陷，以及缺陷类型
    if image is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, CFG.AD_NG, 0

    return CFG.RESULT_OK, msg, CFG.AD_GOOD, 0


def test_detection():
    image_path = "traindata/train/good/142032458.jpg"
    temp_path = "traindata/train/good/142847626.jpg"

    res = isDefect_byPatchCompare(image_path, temp_path,  0.9)
    print(res)


if __name__ == '__main__':
    test_detection()

