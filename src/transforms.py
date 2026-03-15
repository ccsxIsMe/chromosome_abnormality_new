from torchvision import transforms


def build_train_transform(image_size=300):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485],
                             std=[0.229, 0.229, 0.229]),
    ])


def build_val_transform(image_size=300):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485],
                             std=[0.229, 0.229, 0.229]),
    ])