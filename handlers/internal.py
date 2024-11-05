from PIL import Image, ImageChops, ImageEnhance
import uuid
import os

_tempFolder = 'temp'
__all__ = []

def Difference(image: Image, quality: int) -> Image:
    folder = os.path.join(_tempFolder, str(uuid.uuid4()))
    os.mkdir(folder)
    path = os.path.join(folder, "resaved.jpg")
    image.save(path, "JPEG", quality=quality)
    resavedImage = Image.open(path)
    difference = ImageChops.difference(image, resavedImage)
    os.remove(path)
    os.rmdir(folder)
    return difference


def Enhance(image: Image, extrema: int) -> Image:
    scale = 255 / extrema
    image = ImageEnhance.Brightness(image).enhance(scale)
    return image
