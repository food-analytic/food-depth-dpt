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
        is_train,
        dataset_path,
        image_size=(384, 384),
        excluded_files=[],
    ):

        self.is_train = is_train
        self.dataset_path = dataset_path

        if self.is_train:
            self.filepath = os.path.join(
                self.dataset_path, "dish_ids/splits/depth_train_ids.txt"
            )
        else:
            self.filepath = os.path.join(
                self.dataset_path, "dish_ids/splits/depth_test_ids.txt"
            )

        with open(self.filepath) as infile:
            file_list = [file.strip() for file in infile.readlines()]

        legit_files = []

        for file in file_list:
            depth_image_path = os.path.join(
                self.dataset_path, "realsense_overhead", file, "depth_raw.png"
            )
            rgb_image_path = os.path.join(
                self.dataset_path, "realsense_overhead", file, "rgb.png"
            )

            if (
                os.path.exists(depth_image_path)
                and os.path.exists(rgb_image_path)
                and file not in excluded_files
            ):
                legit_files.append(file)

        self.files = legit_files

        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.augment = A.Compose(
            [
                A.Blur(blur_limit=5, p=0.5),
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Flip(p=0.5),
            ],
            additional_targets={"depth": "mask"},
        )

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
        rgb_image = rgb_image / 255.0
        rgb_image = rgb_image.astype(np.float32)

        depth_image = depth_image / 10000.0
        inverse_depth = 1 / (depth_image + 1e-8)
        inverse_depth = inverse_depth.astype(np.float32)

        mask = (depth_image > 0).astype(int)

        if self.is_train:
            augmented = self.augment(image=rgb_image, depth=inverse_depth, mask=mask)
            rgb_image, inverse_depth, mask = (
                augmented["image"],
                augmented["depth"],
                augmented["mask"],
            )

        transformed_image = self.transform(
            {"image": rgb_image, "depth": inverse_depth, "mask": mask}
        )

        return (
            transformed_image["image"],
            transformed_image["depth"],
            transformed_image["mask"],
        )

    def __len__(self):
        return len(self.files)
