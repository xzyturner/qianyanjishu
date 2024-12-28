# filename  图片路径
# ori_shape 这个不知道是啥的感觉用不上
# img_shape (800 1500 3)
# lidar2img lidar2img   局部-》相机-》img
# pad_shape   (800, 1500, 3)
# scale_factor  1   voxel缩放尺度
# pc_range  [-150, -20，-150，150，100，150]   我们的xz是平面 然后y轴朝下面的   目前的voxel_size是1
# occ_size  [occ_size] [300,120,300]
# occ_path
import pickle
# img   6个
import re

# gt occ  N * 4
from datetime import datetime


'''
相机0
1.00000000 0.00000000 0.00000000 0.00000000
0.00000000 0.76604444 -0.64278761 0.00000000
0.00000000 0.64278761 0.76604444 0.00000000
0.00000000 0.00000000 0.00000000 1.00000000
相机180
-1.00000000 -0.00000000 0.00000000 -0.00000000
0.00000000 0.76604444 0.64278761 0.00000000
-0.00000000 0.64278761 -0.76604444 -0.00000000
0.00000000 0.00000000 0.00000000 1.00000000
相机270
0.00000000 -0.00000000 1.00000000 0.00000000
0.64278761 0.76604444 -0.00000000 0.00000000
-0.76604444 0.64278761 0.00000000 -0.00000000
0.00000000 0.00000000 0.00000000 1.00000000
相机90
0.00000000 0.00000000 -1.00000000 -0.00000000
-0.64278761 0.76604444 -0.00000000 0.00000000
0.76604444 0.64278761 0.00000000 0.00000000
0.00000000 0.00000000 0.00000000 1.00000000
bottom
1.00000000 0.00000000 0.00000000 0.00000000
-0.00000000 0.00000000 -1.00000000 -0.00000000
0.00000000 1.00000000 0.00000000 0.00000000
0.00000000 0.00000000 0.00000000 1.00000000
相机内参矩阵
[[750.   0. 750.]
 [  0. 400. 400.]
 [  0.   0.   1.]]
'''

import os

import numpy as np
from PIL import Image

lidar2cams = [
    np.array([[1., 0., 0., 0.],
              [0., -0.6427876, -0.7660444, 0.],
              [0., 0.7660444, -0.6427876, 0.],
              [0., 0., 0., 1.]])
    ,
    np.array([[-1., 0., 0., 0.],
              [0., 0.6427876, 0.7660444, 0.],
              [0., 0.7660444, -0.6427876, 0.],
              [0., 0., 0., 1.]])
    ,
    np.array([[0., -1., 0., 0.],
              [0.7660444, 0., -0.6427876, 0.],
              [0.6427876, 0., 0.7660444, 0.],
              [0., 0., 0., 1.]])
    ,
    np.array([[0., 1., 0., 0.],
              [-0.7660444, 0., 0.6427876, 0.],
              [-0.6427876, 0., -0.7660444, 0.],
              [0., 0., 0., 1.]])
    ,
    np.array([[1., 0., 0., 0.],
              [0., 0., 1., 0.],
              [0., -1., 0., 0.],
              [0., 0., 0., 1.]])
]
# lidar2cams = [
#     np.array([[1.00000000, 0.00000000, 0.00000000, 0.00000000],
#               [0.00000000, 0.76604444, -0.64278761, 0.00000000],
#               [0.00000000, 0.64278761, 0.76604444, 0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[-1.00000000, -0.00000000, 0.00000000, -0.00000000],
#               [0.00000000, 0.76604444, 0.64278761, 0.00000000],
#               [-0.00000000, 0.64278761, -0.76604444, -0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[0.00000000, -0.00000000, 1.00000000, 0.00000000],
#               [0.64278761, 0.76604444, -0.00000000, 0.00000000],
#               [-0.76604444, 0.64278761, 0.00000000, -0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[0.00000000, 0.00000000, -1.00000000, -0.00000000],
#               [-0.64278761, 0.76604444, -0.00000000, 0.00000000],
#               [0.76604444, 0.64278761, 0.00000000, 0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[1.00000000, 0.00000000, 0.00000000, 0.00000000],
#               [-0.00000000, 0.00000000, -1.00000000, -0.00000000],
#               [0.00000000, 1.00000000, 0.00000000, 0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
# ]
intrinsic = \
    np.array([[750, 0, 750],
              [0, 400, 400.],
              [0, 0., 1.]])


def sort_files(files):
    """
    对文件名列表进行排序，使其按照数字顺序排列。

    :param files: 文件名列表
    :return: 排序后的文件名列表
    """

    # 使用正则表达式提取文件名前的数字部分
    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')

    # 根据提取到的数字部分进行排序
    sorted_files = sorted(files, key=extract_number)
    return sorted_files


def combine_matrices(rigid_matrix, intrinsic_matrix):
    """
    组合4x4刚性变换矩阵和3x3内参矩阵，生成新的4x4矩阵。

    :param rigid_matrix: 4x4 刚性变换矩阵
    :param intrinsic_matrix: 3x3 内参矩阵
    :return: 4x4 组合后的矩阵
    """

    # 确保输入矩阵的形状正确
    assert rigid_matrix.shape == (4, 4), "刚性变换矩阵必须是4x4的"
    assert intrinsic_matrix.shape == (3, 3), "内参矩阵必须是3x3的"

    # 将内参矩阵扩展为4x4矩阵
    intrinsic_matrix_4x4 = np.eye(4)
    intrinsic_matrix_4x4[:3, :3] = intrinsic_matrix

    # 计算新的4x4矩阵
    combined_matrix = np.dot(intrinsic_matrix_4x4, rigid_matrix)

    return combined_matrix


def get_image_size_and_channels(image_path):
    """
    读取给定的PNG图片路径并返回图片的尺寸和三通道（RGB）的尺寸。

    :param image_path: 图片的文件路径
    :return: 一个包含图片宽度、高度和三通道尺寸的列表
    """
    with Image.open(image_path) as img:
        width, height = img.size
    # 获取图片的通道信息
    return [height, width, 3]


def tofilename_imgshape_pcrange_occsize_lidar2img_occpath(path):
    file_fold_list = os.listdir(path)
    datas = []
    lists = sort_files(file_fold_list)
    occ_paths_root = path.replace("imgs", "points")
    for fold in lists:
        name_path = path + fold
        file_name = []
        img_shape = []
        pc_range = []
        occ_size = []
        # occ_path = []
        lidar2img = []
        lidar2cam = []
        cam_intrinsic = []
        # 0  180 270 90 bottom
        files = os.listdir(name_path)
        for num in files:
            file_name.append(name_path + "//" + num)
            img_shape.append(get_image_size_and_channels(name_path + "//" + num))
            pc_range.append(np.array([800, 1500, 3]))

        pc_range.append(np.array([-150, -20, -150, 150, 100, 150]))
        occ_size.append(np.array([300, 120, 300]))

        for i in lidar2cams:
            lidar2img.append(combine_matrices(i, intrinsic))
            lidar2cam.append(i)
            cam_intrinsic.append(intrinsic)
        occ_path = occ_paths_root+str(fold)+".npy"
        data = {
            "img_filename": file_name,
            "lidar2img": lidar2img,
            "occ_path": occ_path,
            "cam_intrinsic": cam_intrinsic,
            "lidar2cam": lidar2cam
            # "timestamp":int(datetime.now().timestamp()*1_000_000)
        }
        datas.append(data)
    # location_path = r"E:\project\urbanbisFly\photos\scene\location\\"
    location_root = location_path+str(number)+"pos_ori.txt"
    with open(location_root, 'r') as file:
        lines = file.readlines()
        assert len (lines) == len(datas), "长度不一致"
        i = 0
        for line in lines:
            line = line.strip()
            line_lists = line.split(",")
            datas[i]["timestamp"] = int(float(line_lists[-1]))
            datas[i]["pos"] = line_lists[0]+","+line_lists[1]+","+line_lists[2]
            datas[i]["ori"] = line_lists[3] +","+ line_lists[4] +","+ line_lists[5]
            i=i+1
            # print(line.strip())
    return datas
if __name__ == '__main__':
    number = "four"
    location_path = "/mnt/d/project/urbanbisFly/photos/scene//location//"
    pkl_file_path = '/mnt/d/project/SurroundOcc-main/data/test.pkl'
    train = ["three","four","five","six"]
    val = ["two"]
    test = ["one"]
    datas = []
    for number in test:
        filename_path_root = "/mnt/d/project/urbanbisFly/photos/scene/imgs//" + number + "//"

        # location_path = r"E:\project\urbanbisFly\photos\scene\location\\"
        # pkl_file_path = r'E:\project\urbanbisFly\photos\scene\pkl\data.pkl'
        # filename_path_root = r"E:\project\urbanbisFly\photos\scene\\imgs\\" + number + "\\"
        datas.append(tofilename_imgshape_pcrange_occsize_lidar2img_occpath(path=filename_path_root))
    data_real = []
    for lists in datas:
        for i in lists:
            data_real.append(i)
    datas = {
        'infos': data_real,
        'metadata': {'version': 'v1.0-trainval'}
    }

    # 检查文件是否存在，如果不存在则创建并保存数据
    if not os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(datas, file)
        print(f"data save '{pkl_file_path}'")
    else:
        os.remove(pkl_file_path)
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(datas, file)
        print(f"data save '{pkl_file_path}'")
    print(11)
    # with open(pkl_file_path, 'rb') as file:
    #     loaded_data = pickle.load(file)

    # 访问加载的数据
    # print("加载的文件名列表：", loaded_data['img_filename'])
    # # print("加载的图片尺寸：", loaded_data['img_shapes'])
    # # print("加载的点云范围：", loaded_data['pc_range'])
    # # print("加载的占用栅格大小：", loaded_data['occ_size'])
    # # print("加载的填充尺寸：", loaded_data['pad_shape'])
    # print("加载的LiDAR到图像矩阵：", loaded_data['lidar2img'])
    # print("加载的占用栅格路径：", loaded_data['occ_path'])
