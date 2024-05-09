import numpy as np
import mmcv

from mmdet.datasets.builder import PIPELINES
from numpy import random

@PIPELINES.register_module(force=True)
class Normalize3D(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = [mmcv.imnormalize(
                img, self.mean, self.std, self.to_rgb) for img in results[key]]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module(force=True)
class PadMultiViewImages(object):
    """Pad multi-view images and change intrinsics
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    If set `change_intrinsics=True`, key 'cam_intrinsics' and 'ego2img' will be changed.

    Args:
        size (tuple, optional): Fixed padding size, (h, w).
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
        change_intrinsics (bool): whether to update intrinsics.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0, change_intrinsics=False):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

        self.change_intrinsics = change_intrinsics

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        original_shape = [img.shape for img in results['img']]

        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = [mmcv.impad(
                    img, shape=self.size, pad_val=self.pad_val) for img in results[key]]
            elif self.size_divisor is not None:
                padded_img = [mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val) for img in results[key]]
            results[key] = padded_img

        if self.change_intrinsics:
            post_intrinsics, post_ego2imgs = [], []
            for img, oshape, cam_intrinsic, ego2img in zip(results['img'], \
                    original_shape, results['cam_intrinsics'], results['ego2img']):
                scaleW = img.shape[1] / oshape[1]
                scaleH = img.shape[0] / oshape[0]

                rot_resize_matrix = np.array([ 
                                        [scaleW, 0,      0,    0],
                                        [0,      scaleH, 0,    0],
                                        [0,      0,      1,    0],
                                        [0,      0,      0,    1]])
                post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
                post_ego2img = rot_resize_matrix @ ego2img
                post_intrinsics.append(post_intrinsic)
                post_ego2imgs.append(post_ego2img)
        
            results.update({
                'cam_intrinsics': post_intrinsics,
                'ego2img': post_ego2imgs,
            })


        results['img_shape'] = [img.shape for img in padded_img]
        results['img_fixed_size'] = self.size
        results['img_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        repr_str += f'change_intrinsics={self.change_intrinsics})'

        return repr_str


@PIPELINES.register_module(force=True)
class ResizeMultiViewImages(object):
    """Resize mulit-view images and change intrinsics
    If set `change_intrinsics=True`, key 'cam_intrinsics' and 'ego2img' will be changed

    Args:
        size (tuple, optional): resize target size, (h, w).
        change_intrinsics (bool): whether to update intrinsics.
    """
    def __init__(self, size=None, scale=None, change_intrinsics=True):
        self.size = size
        self.scale = scale
        assert size is None or scale is None
        self.change_intrinsics = change_intrinsics

    def __call__(self, results:dict):

        new_imgs, post_intrinsics, post_ego2imgs = [], [], []

        for img,  cam_intrinsic, ego2img in zip(results['img'], \
                results['cam_intrinsics'], results['ego2img']):
            if self.scale is not None:
                h, w = img.shape[:2]
                target_h = int(h * self.scale)
                target_w = int(w * self.scale)
            else:
                target_h = self.size[0]
                target_w = self.size[1]
            
            tmp, scaleW, scaleH = mmcv.imresize(img,
                                                # NOTE: mmcv.imresize expect (w, h) shape
                                                (target_w, target_h),
                                                return_scale=True)
            new_imgs.append(tmp)

            rot_resize_matrix = np.array([
                [scaleW, 0,      0,    0],
                [0,      scaleH, 0,    0],
                [0,      0,      1,    0],
                [0,      0,      0,    1]])
            post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
            post_ego2img = rot_resize_matrix @ ego2img
            post_intrinsics.append(post_intrinsic)
            post_ego2imgs.append(post_ego2img)

        results['img'] = new_imgs
        results['img_shape'] = [img.shape for img in new_imgs]
        if self.change_intrinsics:
            results.update({
                'cam_intrinsics': post_intrinsics,
                'ego2img': post_ego2imgs,
            })

        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'change_intrinsics={self.change_intrinsics})'

        return repr_str
    

@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            # if random.randint(2):
            #     img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str