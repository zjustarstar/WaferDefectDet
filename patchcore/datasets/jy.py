import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class JYDataset(Dataset):
    def __init__(self, data_path, resize=366, imagesize=320, is_train=True):
        super(JYDataset, self).__init__()
        filenames = sorted(os.listdir(data_path))
        self.images_path = [os.path.join(data_path, filename) for filename in filenames]
        self.transforms_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms_img(image)

        return {"image": image}

    def __len__(self):
        return len(self.images_path)
