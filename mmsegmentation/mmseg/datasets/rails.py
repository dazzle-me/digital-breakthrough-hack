import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

classes = ('background', 'side_rails', 'main_rails', 'train')
palette = ([0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255])

@DATASETS.register_module()
class RailsDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None
