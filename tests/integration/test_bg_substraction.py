import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rpca import RobustPCA


class TestBGSubstraction(unittest.TestCase):
    def test_bg_substraction(self):
        data_dir = Path(__file__).parent / "data"
        data_files = sorted(data_dir.glob("*"))

        img_array = []
        img_shape = np.array(Image.open(data_files[0])).shape

        # Load images
        for f in data_files:
            img_array.append(np.array(Image.open(f)).flatten())
        img_array = np.array(img_array).astype(float).T

        rpca = RobustPCA(n_components=2)

        rpca.fit(img_array)

        img_bg = (rpca.low_rank_[:, 0] + rpca.mean_[0]).reshape(img_shape)
        img_fg = (rpca.sparse_[:, 0] + rpca.mean_[0]).reshape(img_shape)
        img_bg = np.clip(img_bg, 0, 255)
        img_fg = np.clip(img_fg, 0, 255)

        bg = Image.fromarray(img_bg.astype(np.uint8), mode="L")
        fg = Image.fromarray(img_fg.astype(np.uint8), mode="L")

        target_bg = Image.open(Path(__file__).parent / "target" / "bg.png")
        target_fg = Image.open(Path(__file__).parent / "target" / "fg.png")

        self.assertTrue(np.allclose(np.array(bg), np.array(target_bg)))
        self.assertTrue(np.allclose(np.array(fg), np.array(target_fg)))
