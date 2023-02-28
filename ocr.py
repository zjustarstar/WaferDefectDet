from paddleocr import PaddleOCR, draw_ocr
import PIL.Image as Image
import config as CFG


# cls_image_shape='3, 770, 500',
ocr = PaddleOCR(use_angle_cls=True, cls_thresh=0.5,
                cls_model_dir='whl\\cls\\ch_ppocr_mobile_v2.0_cls_infer',
                det_model_dir='whl\\det\\en\\en_PP-OCRv3_det_infer',
                rec_model_dir='whl\\rec\\en\\en_PP-OCRv3_rec_infer',
                lang='en')


# 版号读取
def get_rectileID(img_file):
    result = ocr.ocr(img_file, cls=True)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)

    msg = "No text found"
    rslt = CFG.RESULT_FAIL
    result = result[0]
    if len(result) > 0:
        # list[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],...]
        boxes = [line[0] for line in result]
        # list['aaa','bbb'...]
        txts = [line[1][0] for line in result]
        # list[0.2,0.2...]
        scores = [line[1][1] for line in result]
        msg = "text found"
        rslt = CFG.RESULT_OK

    return rslt, msg, boxes, txts, scores


if __name__ == '__main__':
    img_file = "testimg/ocr/wafernum2.bmp"

    _, _, boxes, txts, scores = get_rectileID(img_file)
    image = Image.open(img_file).convert('RGB')
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.show("res")
    im_show.save('ocr_result.jpg')
