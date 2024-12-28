# import open3d as o3d
import mmcv
import numpy as np
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from PIL import Image
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os
import cv2


@PIPELINES.register_module()
class LoadOccupancy_urb_airsim(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class.
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic

    def __call__(self, results):

        data = np.loadtxt(results['occ_path'], delimiter=',', skiprows=1)
        points = data[:, :4]

        occ = np.array(points)
        occ = occ.astype(np.float32)

        # class 0 is 'ignore' class
        # if self.use_semantic:
        #     occ[..., 3][occ[..., 3] == 0] = 255
        # else:
        #     occ = occ[occ[..., 3] > 0]
        #     occ[..., 3] = 1
        # occ[:,0] = ((occ[:,0]*2-192)+192)/4
        # occ[:, 1] = ((occ[:,1]*2-0)+128)/4
        # occ[:, 2] = ((occ[:,0]*2-192)+192)/4
        results['gt_occ'] = occ

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage_Airsim:
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
            if random.randint(0,2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(0,2)
            if mode == 1:
                if random.randint(0,2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(0,2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(0,2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(0,2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # # randomly swap channels
            # if random.randint(0,2):
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
@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_Airsim(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        filename = filename.strip("[]").replace("'", "").split(", ")
        # print("test3:")
        # for i in filename:
        #     print("file:")
        #     print("color")
        #     print(print(self.color_type))
        #     print(i)
        #     print("a")
        #     a = mmcv.imread(i, self.color_type)
        #     print(a.shape)
        #     print("b")
        #     b = mmcv.imread(i)
        #     print(b.shape)
        # img is of shape (h, w, c, num_views)

        img = np.stack(
            [cv2.imread(name) for name in filename], axis=-1)
        # [mmcv.imread(name, self.color_type) for name in filename], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class Readmetas(object):

    def __call__(self, results):
        results['lidar2img'] = np.load(results['lidar2img'])
        results['intrinsic'] = np.load(results['intrinsic'])
        results['ego2world'] = np.load(results['ego2world'])
        results['occ_size'] = np.array(list(map(int, results['occ_size'].strip('[]').split(','))))
        results["pc_range"] = np.array(list(map(int, results['pc_range'].strip('[]').split(','))))
        # results['filename'] = np.array(results['filename'].split(';'))
        lidar2img = np.array(
            [
                [
                    [750.00000000, 649.51905284, 375.00000000, 0.00000000],
                    [0.00000000, 750.91016151, -500.61455166, 0.00000000],
                    [0.00000000, 0.86602540, 0.50000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ],
                [
                    [-375.00000000, 649.51905284, 750.00000000, 0.00000000],
                    [500.61455166, 750.91016151, -0.00000000, 0.00000000],
                    [-0.50000000, 0.86602540, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ],
                [
                    [375.00000000, 649.51905284, -750.00000000, 0.00000000],
                    [-500.61455166, 750.91016151, -0.00000000, 0.00000000],
                    [0.50000000, 0.86602540, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ],
                [
                    [750.00000000, 750.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 400.00000000, -809.00000000, 0.00000000],
                    [0.00000000, 1.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ],
                [
                    [-750.00000000, 649.51905284, -375.00000000, 0.00000000],
                    [0.00000000, 750.91016151, 500.61455166, 0.00000000],
                    [-0.00000000, 0.86602540, -0.50000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ]
            ]
        )
        # lidar2img = np.array(
        #     [
        #         [
        #             [750.00000000, 649.51905284, 375.00000000, 3750.00000000],
        #             [0.00000000, 750.91016151, -500.61455166, -5006.14551662],
        #             [0.00000000, 0.86602540, 0.50000000, 5.00000000],
        #             [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        #         ],
        #         [
        #             [-375.00000000, 649.51905284, 750.00000000, 7500.00000000],
        #             [500.61455166, 750.91016151, -0.00000000, -0.00000000],
        #             [-0.50000000, 0.86602540, 0.00000000, 0.00000000],
        #             [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        #         ],
        #         [
        #             [375.00000000, 649.51905284, -750.00000000, -7500.00000000],
        #             [-500.61455166, 750.91016151, -0.00000000, -0.00000000],
        #             [0.50000000, 0.86602540, 0.00000000, 0.00000000],
        #             [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        #         ],
        #         [
        #             [750.00000000, 750.00000000, 0.00000000, 0.00000000],
        #             [0.00000000, 400.00000000, -809.00000000, -8090.00000000],
        #             [0.00000000, 1.00000000, 0.00000000, 0.00000000],
        #             [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        #         ],
        #         [
        #             [-750.00000000, 649.51905284, -375.00000000, -3750.00000000],
        #             [0.00000000, 750.91016151, 500.61455166, 5006.14551662],
        #             [-0.00000000, 0.86602540, -0.50000000, -5.00000000],
        #             [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        #         ]
        #     ]
        # )
        results['lidar2img'] = lidar2img
        return results


@PIPELINES.register_module()
class CustomCollect3D_urban(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    # def __init__(self,
    #              keys,
    #              meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img' ,
    #                         'depth2img', 'cam2img', 'pad_shape',
    #                         'scale_factor', 'flip', 'pcd_horizontal_flip',
    #                         'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
    #                         'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
    #                         'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
    #                         'transformation_3d_flow', 'scene_token',
    #                         'can_bus', 'pc_range', 'occ_size', 'occ_path', 'lidar_token'
    #                         )):
    def __init__(self,
                 keys,
                 meta_keys=('TimeStamp','occ_size','POS_X','POS_Y','POS_Z','Q_W','Q_X','Q_Y','Q_Z','Roll','Pitch',
         'Yaw','depth','lidar2img','intrinsic','ego2world','previous_time','next_time',
         'images_root','x_range','y_range','z_range','pc_x_range','pc_y_range','pc_z_range',
         'img_filename','occ_path','img','img_shape','ori_shape','pad_shape','scale_factor',
         'img_norm_cfg','gt_occ','pad_size_divisor','pc_range','filename')
                 ):

        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        data = {}
        img_metas = {}

        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'