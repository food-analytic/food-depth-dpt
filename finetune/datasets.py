import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import albumentations as A
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


class Nutrition5k(Dataset):
    def __init__(
        self,
        split,
        dataset_path,
        image_size=(384, 384),
    ):

        self.split = split
        self.dataset_path = dataset_path

        if self.split == "train":
            self.filepath = "finetune/splits/train.txt"
        elif self.split == "val":
            self.filepath = "finetune/splits/val.txt"
        elif self.split == "test":
            self.filepath = "finetune/splits/test.txt"
        else:
            raise ValueError("Invalid split name.")

        with open(self.filepath) as infile:
            self.files = [file.strip() for file in infile.readlines()]

        self.augment = A.Compose(
            [
                A.RandomCrop(480, 480, p=1.0),
                A.Flip(p=0.5),
                A.Blur(blur_limit=5, p=0.5),
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomGamma(p=0.5),
            ],
            additional_targets={"depth": "mask"},
        )

        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.transform = Compose(
            [
                Resize(
                    image_size[0],
                    image_size[1],
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )

    def __getitem__(self, index):
        file = self.files[index]

        depth_image_path = os.path.join(
            self.dataset_path, "realsense_overhead", file, "depth_raw.png"
        )
        rgb_image_path = os.path.join(
            self.dataset_path, "realsense_overhead", file, "rgb.png"
        )

        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        rgb_image = cv2.imread(rgb_image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        depth_image = depth_image / 10000.0
        inverse_depth = 1 / (depth_image + 1e-8)
        inverse_depth = inverse_depth.astype(np.float32)

        mask = (depth_image > 0).astype(int)

        if self.split == "train":
            augmented = self.augment(image=rgb_image, depth=inverse_depth, mask=mask)
            rgb_image, inverse_depth, mask = (
                augmented["image"],
                augmented["depth"],
                augmented["mask"],
            )

        rgb_image = rgb_image / 255.0
        rgb_image = rgb_image.astype(np.float32)

        transformed_image = self.transform(
            {"image": rgb_image, "depth": inverse_depth, "mask": mask}
        )

        rgb_image = transformed_image["image"]
        inverse_depth = transformed_image["depth"]
        mask = transformed_image["mask"]

        return rgb_image, inverse_depth, mask

    def __len__(self):
        return len(self.files)
