####### Copied from SegmentAndComplete with minor modifications to add rain #######
####### https://arxiv.org/pdf/2112.04532 #######


import coco.coco_transforms as T


class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, mean=(123., 117., 104.)):
        if data_augmentation == 'hflip':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
        elif data_augmentation == 'ssd':
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=list(mean)),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
        elif data_augmentation == 'ssdlite':
            self.transforms = T.Compose([
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)
