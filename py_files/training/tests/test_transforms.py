from PIL import Image
from src.data.build_features import define_transforms

def test_transforms_turn_a_full_size_photo_into_3x64x64():
    photo = Image.new("RGB", (1600, 768))
    train_transforms, val_transforms = define_transforms()

    assert train_transforms(photo).shape == (3, 64, 64)
    assert val_transforms(photo).shape == (3, 64, 64)