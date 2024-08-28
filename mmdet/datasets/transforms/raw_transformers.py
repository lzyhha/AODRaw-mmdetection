from mmdet.registry import TRANSFORMS
from mmcv.transforms import LoadImageFromFile
import mmcv
import numpy as np
from typing import Optional
import torch
import enum


@TRANSFORMS.register_module()
class LoadImageFromRAW(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 gamma: float = 1.0, 
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.gamma = gamma
        super().__init__(to_float32, color_type, imdecode_backend, file_client_args, ignore_empty, backend_args=backend_args)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        img = np.load(filename)
        img = np.transpose(img, (1, 2, 0))
        img = mmcv.rgb2bgr(img)

        if self.gamma != 1.0:
            img = img / 255.0
            img = np.power(img, 1 / self.gamma)
            img = img * 255.0

        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    From https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


class Debayer3x3(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 convolutions.

    From https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel reflection padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer3x3, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    [0, 0.25, 0],
                    [0.25, 0, 0.25],
                    [0, 0.25, 0],

                    [0.25, 0, 0.25],
                    [0, 0, 0],
                    [0.25, 0, 0.25],

                    [0, 0, 0],
                    [0.5, 0, 0.5],
                    [0, 0, 0],

                    [0, 0.5, 0],
                    [0, 0, 0],
                    [0, 0.5, 0],
                ]
            ).view(4, 1, 3, 3),
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        c = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        c = torch.cat((c, x), 1)  # Concat with input to give identity kernel Bx5xHxW

        rgb = torch.gather(
            c,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand in batch is faster than repeat
        )
        return rgb

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 2],  # pixel is R,G1
                [3, 1],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [1, 3],  # pixel is R,G1
                [2, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)


def _unpack_bayer(x):
    c, h, w = x.shape
    assert c == 4
    H = h * 2
    W = w * 2
    out = np.zeros((H, W), dtype = x.dtype)

    out[0:H:2, 0:W:2] = x[0, :, :]
    out[0:H:2, 1:W:2] = x[1, :, :]
    out[1:H:2, 1:W:2] = x[2, :, :]
    out[1:H:2, 0:W:2] = x[3, :, :]
    return out


@TRANSFORMS.register_module()
class LoadImageFromRAWDemosaicing(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 gamma: float = 1.0, 
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.gamma = gamma
        self.debayer = Debayer3x3()
        super().__init__(to_float32, color_type, imdecode_backend, file_client_args, ignore_empty, backend_args=backend_args)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        img = np.load(filename)
        img = _unpack_bayer(img)
        
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = self.debayer(img)
        img = img.squeeze(0).cpu().numpy()

        img = np.transpose(img, (1, 2, 0))
        img = mmcv.rgb2bgr(img)

        if self.gamma != 1.0:
            img = img / 255.0
            img = np.power(img, 1 / self.gamma)
            img = img * 255.0

        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
