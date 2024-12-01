# High Frequency Noise

from PIL import Image
import numpy as np
from . import internal

__all__ = ["Prepare"]


def Prepare(pathToImage: str, quality: int, size: tuple[int, int]) -> np.ndarray:
    image = HFN(pathToImage, quality)
    image = image.resize(size)
    data = np.array(image).flatten() / 255
    return data


def HFN(pathToImage: str, quality: int) -> Image:
    image = Image.open(pathToImage).convert("L")
    image = internal.Difference(image, quality)
    image = Enhance(image)
    return image


def Enhance(image: Image) -> Image:
    extrema = max(image.getextrema())
    if extrema == 0:
        extrema = 1
    image = internal.Enhance(image, extrema)
    return image
