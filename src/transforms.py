from torchvision import transforms


def build_train_transform(image_size=300):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.229, 0.229],
        ),
    ])


def build_style_transform(image_size=300):
    """
    Stronger style perturbation branch for single-domain generalization.
    Keep geometry moderate; emphasize illumination / contrast / blur / slight affine perturbation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.05,
                hue=0.02,
            )
        ], p=0.9),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=12)
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=3,
            )
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.229, 0.229],
        ),
    ])


def build_val_transform(image_size=300):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.229, 0.229],
        ),
    ])