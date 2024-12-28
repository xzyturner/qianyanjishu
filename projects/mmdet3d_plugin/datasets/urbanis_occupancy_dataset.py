import pickle
import tempfile
from os import path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from mmdet3d.datasets import NuScenesDataset, Custom3DDataset
from mmdet.datasets import DATASETS

from projects.mmdet3d_plugin.datasets import Det3DDataset


# def load_image(image_path):
#     # 使用PIL库读取图片
#     image = Image.open(image_path)
#     # 转换为RGB格式（确保三通道）
#     image = image.convert('RGB')
#     # 转换为numpy数组
#     image = np.array(image)
#     # 转换为tensor并将通道维度从最后移到第一个位置
#     image = torch.tensor(image).permute(2, 0, 1)
#     return image

@DATASETS.register_module()
class CustomUrbOccDataset(NuScenesDataset):
    def __init__(self, pkl_file_path, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, *args,
                 **kwargs):
        # 加载PKL文件中的数据
        super().__init__(*args, **kwargs)
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self._set_group_flag()
        self.pkl_file_path = pkl_file_path
        # with open(pkl_file_path, 'rb') as file:
            # self.data = pickle.load(file)
        # 获取数据长度
        # with open(self.pkl_file_path, 'rb') as file:
        #     data = pickle.load(file)
        #     data = data["infos"]
        # self.length = len(data['img_filename'])
        # print(self.length)

    def __len__(self):
        #这个有个bug调用不了self里面的私参

        with open(self.ann_file, 'rb') as file:
            data = pickle.load(file)
            data = data["infos"]
        return len(data)

    #这两个重写
    def prepare_train_data(self, index):
        with open(self.pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        data = data["infos"]
        filename = data[index]['img_filename']
        occ_size = self.occ_size
        pc_range = self.pc_range
        lidar2img = data[index]['lidar2img']
        occ_path = data[index]['occ_path']
        # print("occ_path",occ_path)
        cam_intrinsic = data[index]['cam_intrinsic']
        lidar2cam = data[index]['lidar2cam']
        # gt_occ = np.fromfile(occ_path, dtype=np.float32).reshape(-1, 4)
        # 返回一个包含所有数据的字典
        input_dict = {
            'img_filename': filename,
            'pc_range': pc_range,
            'occ_size': occ_size,
            'lidar2img': lidar2img,
            'occ_path': occ_path,
            "cam_intrinsic": cam_intrinsic,
            "lidar2cam": lidar2cam
        }
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        # images = []
        # for image_path in filename:
        #     image = load_image(image_path)
        #     images.append(image)
        # # 将所有图片堆叠到一个tensor中
        # images_tensor = torch.stack(images)
        #
        # example = {
        #     'img_metas': img_metas,
        #     'img': images_tensor,
        #     'gt_occ': gt_occ
        # }
    def prepare_test_data(self, index):
        # print("data:")
        # print(self.data)
        with open(self.pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        data = data["infos"]

        filename = data[index]['img_filename']
        print("filename_test:")
        print(filename)
        occ_size = self.occ_size
        pc_range = self.pc_range
        lidar2img = data[index]['lidar2img']
        occ_path = data[index]['occ_path']
        cam_intrinsic = data[index]['cam_intrinsic']
        lidar2cam = data[index]['lidar2cam']
        # gt_occ = np.fromfile(occ_path, dtype=np.float32).reshape(-1, 4)
        # 返回一个包含所有数据的字典
        input_dict = {
            'img_filename': filename,
            'pc_range': pc_range,
            'occ_size': occ_size,
            'lidar2img': lidar2img,
            'occ_path': occ_path,
            "cam_intrinsic": cam_intrinsic,
            "lidar2cam": lidar2cam
        }
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        # 获取每个索引的数据

        if self.test_mode:
            info = self.data_infos[idx]

            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            # file_name  = data["img_filename"]
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]

            results = np.stack(results, axis=0).mean(0)
            mean_ious = []

            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp
                mean_ious.append(tp / union)

            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])


        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict = {'Acc': results[0],
                            'Comp': results[1],
                            'CD': results[2],
                            'Prec': results[3],
                            'Recall': results[4],
                            'F-score': results[5]}

        return results_dict

# # 示例PKL文件路径
# pkl_file_path = '/media/vcc/LinuxData/project/urbanbisFly/photos/scene/pkl/data.pkl'
#
# # 创建Dataset对象
# dataset = CustomDataset(pkl_file_path)
#
# # 创建DataLoader对象
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#
# # 遍历DataLoader
# for i, data in enumerate(dataloader):
#     print(f"Batch {i + 1}:")
#     print(data)
