import pickle
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os import path as osp


def load_image(image_path):
    # 使用PIL库读取图片
    image = Image.open(image_path)
    # 转换为RGB格式（确保三通道）
    image = image.convert('RGB')
    # 转换为numpy数组
    image = np.array(image)
    # 转换为tensor并将通道维度从最后移到第一个位置
    image = torch.tensor(image).permute(2, 0, 1)
    return image


class CustomDataset(Dataset):

    def __init__(self, pkl_file_path, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # 加载PKL文件中的数据
        with open(pkl_file_path, 'rb') as file:
            self.data = pickle.load(file)

        # 获取数据长度
        self.length = len(self.data['filenames'])
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 获取每个索引的数据
        filename = self.data['filenames'][idx]
        img_shape = self.data['img_shapes'][idx]
        pc_range = self.data['pc_range'][idx]
        occ_size = self.data['occ_size'][idx]
        pad_shape = self.data['pad_shape'][idx]
        lidar2img = self.data['lidar2img'][idx]
        occ_path = self.data['occ_path'][idx]
        gt_occ = np.fromfile(occ_path, dtype=np.float32).reshape(-1, 4)
        # 返回一个包含所有数据的字典
        img_metas = {
            'filename': filename,
            'img_shape': img_shape,
            'pc_range': pc_range,
            'occ_size': occ_size,
            'pad_shape': pad_shape,
            'lidar2img': lidar2img,
            'occ_path': occ_path
        }
        images = []
        for image_path in filename:
            image = load_image(image_path)
            images.append(image)
        # 将所有图片堆叠到一个tensor中
        images_tensor = torch.stack(images)

        example = {
            'img_metas': img_metas,
            'img': images_tensor,
            'gt_occ': gt_occ
        }
        return example

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
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

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

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


# 示例PKL文件路径
pkl_file_path = 'C:\\Users\Admin\Desktop\wor\\urbanbisFly\photos\scene\pkl\data.pkl'

# 创建Dataset对象
dataset = CustomDataset(pkl_file_path)

# 创建DataLoader对象
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 遍历DataLoader
for i, data in enumerate(dataloader):
    print(f"Batch {i + 1}:")
    print(data)
