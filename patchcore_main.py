import contextlib
import gc
import glob
import os
from typing import List

import PIL
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import patchcore.sampler
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
from patchcore.datasets.mvtec import MVTecDataset
from patchcore.datasets.jy import JYDataset

img_resize, crop_size = 366, 320


def train_data(ok_image_path: str, weight_save_path) -> str:
    """
    根据正常图片训练模型
    Args:
        ok_image_path: 正常图片的路径，dir
        weight_save_path: 保存权重文件的路径
    Returns: 模型权重文件夹路径
    """
    # 加载数据集
    # source = os.path.split(ok_image_path)[0]
    # classname = os.path.split(ok_image_path)[1]
    # train_dataset = MVTecDataset(source, classname, resize=img_resize, imagesize=crop_size)
    train_dataset = JYDataset(ok_image_path, resize=img_resize, imagesize=crop_size)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device("cpu")
    device_context = (
        torch.cuda.device(f'cuda:{device.index}')
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    with device_context:
        # torch.cuda.empty_cache()

        # 载入模型
        imagesize = train_dataset.imagesize
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.1, device)
        backbone_name = 'wideresnet50'
        layers_to_extract_from_coll = ['layer2', 'layer3']
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name = backbone_name
        nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=8)
        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from_coll,
            device=device,
            input_shape=imagesize,
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patchsize=3,
            anomaly_score_num_nn=5,
            featuresampler=sampler,
            nn_method=nn_method
        )

        # 训练
        patchcore_instance.fit(train_dataloader)
        os.makedirs(weight_save_path, exist_ok=True)
        patchcore_instance.save_to_path(weight_save_path)

    return weight_save_path


def load_model(weight_path):
    device = torch.device("cpu")

    if not (os.path.exists(os.path.join(weight_path, "patchcore_params.pkl"))
            and os.path.exists(os.path.join(weight_path, "nnscorer_search_index.faiss"))):
        return None
    # device_context = (
    #     torch.cuda.device(f'cuda:{device.index}')
    #     if "cuda" in device.type.lower()
    #     else contextlib.suppress()
    # )
    # with device_context:
    # torch.cuda.empty_cache()
    # 加载模型
    gc.collect()
    nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=8)
    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load_from_path(
        load_path=weight_path, device=device, nn_method=nn_method
    )

    return patchcore_instance


def get_single_image_score(image_path: str, patchcore_instance) -> float:
    """
    计算单张图片的异常得分，分数越大，异常的可能性越高
    Args:
        image_path: 图片的路径，可以是 file 或 dir
        weight_path: 模型权重路径，dir

    Returns: 异常分数列表

    """
    transform_img = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    im = PIL.Image.open(image_path)
    if im is None:
        return -1

    im = im.convert("RGB")
    im = transform_img(im)
    im = im.unsqueeze(0)
    score, _ = patchcore_instance.predict(im)

    return float(score[0])


def get_images_score(image_path: str, patchcore_instance):
    """
    计算每张图片的异常得分，分数越大，异常的可能性越高
    Args:
        image_path: 图片的路径，可以是 file 或 dir
        weight_path: 模型权重路径，dir

    Returns: 异常分数列表

    """
    # device = torch.device("cpu")
    # # device_context = (
    # #     torch.cuda.device(f'cuda:{device.index}')
    # #     if "cuda" in device.type.lower()
    # #     else contextlib.suppress()
    # # )
    # # with device_context:
    # # torch.cuda.empty_cache()
    # # 加载模型
    # gc.collect()
    # nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=8)
    # patchcore_instance = patchcore.patchcore.PatchCore(device)
    # patchcore_instance.load_from_path(
    #     load_path=weight_path, device=device, nn_method=nn_method
    # )

    # ------------------------------------------------------------------------
    images = glob.glob(os.path.join(image_path, '*'))

    transform_img = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    scores = []
    for image in images:
        with PIL.Image.open(image) as im:
            im = im.convert("RGB")
            im = transform_img(im)
            im = im.unsqueeze(0)
            score, _ = patchcore_instance.predict(im)
            scores.append([image, score[0]])

    return scores


if __name__ == '__main__':
    # weight_path = train_data("traindata/OK", './models')
    patchcore_instance = load_model("./models/aa")
    print("load model ok")
    # scores = get_images_score("./traindata/NG", patchcore_instance)
    score = get_single_image_score('./testimg/test.jpg', patchcore_instance)
    print(score)


def cv2_img_is_defect(cv2_img, patchcore_model):
    transform_img = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    im = PIL.Image.fromarray(cv2_img)
    if im is None:
        return -1

    im = im.convert("RGB")
    im = transform_img(im)
    im = im.unsqueeze(0)
    score, _ = patchcore_model.predict(im)

    return 1 if float(score[0]) > 2.0 else 0
