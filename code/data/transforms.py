import albumentations as albu


def get_transform(size, transform_type="weak", min_visibility=0):
    bbox_params = {
        'format': 'coco',
        'min_visibility': min_visibility,
        'label_fields': ['category_id']
    }

    augs = {'strong': albu.Compose([albu.Resize(size, size),
                                    albu.HorizontalFlip(),
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=.4),
                                    # albu.OpticalDistortion(),
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.4),
                                    albu.HueSaturationValue(p=0.3)
                                    ], bbox_params=bbox_params),
            'weak': albu.Compose([albu.Resize(size, size),
                                  # albu.HorizontalFlip(),
                                  ], bbox_params=bbox_params),
            }

    aug_fn = augs[transform_type]
    normalize = albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    pipeline = albu.Compose([aug_fn, normalize])

    return pipeline
