from copy import deepcopy
import os
import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from utils.datasets import (
    InfiniteDataLoader,
    augment_hsv,
    letterbox,
    load_image,
    load_mosaic,
    load_mosaic9,
    random_perspective,
    pastein,
    load_samples,
)
from utils.datasets import LoadImagesAndLabels
from utils.general import xywhn2xyxy, xyxy2xywh
from utils.torch_utils import torch_distributed_zero_first


def create_dataloader_semi(
    path,
    imgsz,
    batch_size,
    stride,
    opt,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    world_size=1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    mosaic=False
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabelsSemi(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=opt.single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            mosaic=mosaic
        )

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsSemi.collate_fn4
        if quad
        else LoadImagesAndLabelsSemi.collate_fn,
    )
    return dataloader, dataset


# def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
#     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
#     y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
#     y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
#     y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
#     return y


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


import torchvision.transforms as transforms


def build_strong_augmentation():
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    augmentation = []

    # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
    augmentation.append(
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    )
    augmentation.append(transforms.RandomGrayscale(p=0.2))
    augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

    # randcrop_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.RandomErasing(
    #             p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
    #         ),
    #         transforms.RandomErasing(
    #             p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
    #         ),
    #         transforms.RandomErasing(
    #             p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
    #         ),
    #         transforms.ToPILImage(),
    #     ]
    # )
    # augmentation.append(randcrop_transform)

    return transforms.Compose(augmentation)


class LoadImagesAndLabelsSemi(LoadImagesAndLabels):
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        mosaic=True,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        prefix="",
    ):
        super().__init__(
            path,
            img_size,
            batch_size,
            augment,
            hyp,
            rect,
            image_weights,
            cache_images,
            single_cls,
            stride,
            pad,
            prefix,
        )
        self.strong_augmentation = build_strong_augmentation()
        self.mosaic = mosaic

    def __getitem__(self, index):

        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            img: (3, 640, 640)
            labels_out: 6 (index, label, x, y, w, h)
                tensor([[0.00000, 0.00000, 0.38615, 0.39738, 0.06601, 0.06488],
                    [0.00000, 0.00000, 0.94521, 0.35268, 0.03656, 0.08920],
                    [0.00000, 0.00000, 0.51676, 0.60980, 0.05078, 0.05642],
                    [0.00000, 0.00000, 0.90180, 0.67327, 0.12338, 0.12989]]),
            image_path: list[path]
            shapes:(ratio, not int)
        """

        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic

            if random.random() < 0.8:
                img, labels = load_mosaic(self, index)
            else:
                img, labels = load_mosaic9(self, index)

            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp["mixup"]:
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(
                        self, random.randint(0, len(self.labels) - 1)
                    )
                else:
                    img2, labels2 = load_mosaic9(
                        self, random.randint(0, len(self.labels) - 1)
                    )
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
                )

        if self.augment:
            img, labels = random_perspective(
                img,
                labels,
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
            )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if not mosaic:
            # Albumentations
            # img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            # augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        label_class_one_hot = torch.zeros(hyp["nc"])
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # strong augmentations
        img_strong = img.copy()
        augment_hsv(
            img_strong, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"]
        )
        img_strong = Image.fromarray(img_strong.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(img_strong))
        # image_strong_aug = transforms.ToPILImage()(image_strong_aug).convert("RGB")
        image_strong_aug = np.array(image_strong_aug)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        image_weak_aug = img
        image_strong_aug = image_strong_aug[:, :, ::-1].transpose(
            2, 0, 1
        )  # BGR to RGB, to 3x416x416
        image_strong_aug = np.ascontiguousarray(image_strong_aug)

        for ind in labels_out[:, 1].long():
            label_class_one_hot[ind] = 1

        return (
            torch.from_numpy(image_weak_aug) / 255,
            torch.from_numpy(image_strong_aug) / 255,
            labels_out,
            label_class_one_hot.unsqueeze(0),
            self.img_files[index],
        )

    @staticmethod
    def collate_fn(batch):
        img_w, img_s, label, label_class_one_hot, path = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return (
            (torch.stack(img_w, 0), torch.stack(img_s, 0)),
            torch.cat(label, 0),
            torch.cat(label_class_one_hot, 0)
        )

    @staticmethod
    def collate_fn4(batch):
        return super().collate_fn4(batch)
