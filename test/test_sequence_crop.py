import unittest

import numpy as np
import torch

from hocap_annotation.loaders import SequenceLoader


class CropTests(unittest.TestCase):
    def loader(self, crop):
        loader = SequenceLoader.__new__(SequenceLoader)
        loader._num_cams = 1
        loader._device = 'cpu'
        loader._rays = torch.ones((1, 3, 4))
        loader._extr2world = torch.eye(4).unsqueeze(0)
        loader._in_world = True
        loader._crop_lim = crop
        return loader

    def test_no_crop_keeps_positive_finite_depth_only(self):
        loader = self.loader(None)
        _, _, mask = loader._deproject(np.zeros((1, 1, 4, 3), np.float32),
                                     np.array([[[0., 1., np.nan, np.inf]]], np.float32))
        self.assertEqual(mask.tolist(), [[False, True, False, False]])

    def test_world_crop_still_applies(self):
        loader = self.loader([-2., 2., -2., 2., .5, 1.5])
        _, _, mask = loader._deproject(np.zeros((1, 1, 4, 3), np.float32),
                                     np.array([[[0., .25, 1., 2.]]], np.float32))
        self.assertEqual(mask.tolist(), [[False, False, True, False]])


if __name__ == '__main__':
    unittest.main()
