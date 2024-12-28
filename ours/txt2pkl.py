
import pickle
import pandas as pd
import os
import csv
def check(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        # 跳过文件的表头
        header = next(reader)

        # 检查每一行的数据
        for row in reader:
            x, y, z = map(float, row[:3])  # 读取前三列并转化为浮点数

            # 检查是否有任何坐标大于96
            if x > 96 or y > 64 or z > 96:
                print(file_path)
                break
            if x <0 or y<0 or z <0:
                print(file_path)
                break
# pkl_file = '/mnt/d/dataset/urbanbis/2//datas.pkl'

data_name_train  = ['yuehai','lihu']
data_name_val = ["val_yuehai"]
data_name_test = ['test_yuehai']

root_path =  "/mnt/d/dataset/urbanbis//"

train_pkl_file = "/mnt/d/project/SurroundOcc-main/data/urbanbis_test/urbanbis_info_train.pkl"
val_pkl_file = "/mnt/d/project/SurroundOcc-main/data/urbanbis_test/urbanbis_info_val.pkl"
test_pkl_file = "/mnt/d/project/SurroundOcc-main/data/urbanbis_test/urbanbis_info_test.pkl"
# 存储字典对象的列表

train_val_test =['train', 'val', 'test']


for mode in train_val_test:
    if mode == 'train':
        mode_data = data_name_train
    if mode == 'val':
        mode_data = data_name_val
    if mode == 'test':
        mode_data =data_name_test
    dict_list = []
    for map_name in mode_data:

        files = os.listdir(root_path+map_name)
        for fold in files:
            file_path = root_path + "//" + str(map_name)+"//"+fold + "//datas.txt"

            with open(file_path, 'r') as file:
                # 读取第一行，作为字典的键
                keys = file.readline().strip().split(' ')
                # 遍历后续的每一行
                keys = [item.strip() for item in keys]
                keys = [item.replace(',', '') for item in keys]

                for line in file:
                    # 去掉行尾的换行符和空格
                    # line = line.strip()
                    if line:  # 忽略空行
                        # 使用逗号分隔不同的值
                        values = line.split('~')
                        # 将键值对打包成字典
                        d = dict(zip(keys, values))
                        # 将字典添加到列表中
                        # a = d["images_root"].split("imgs")
                        # a = a[0]+"//imgs"+a[1]+"//imgs"+a[2]+"//imgs"+a[3]+"//imgs"+a[4]+"//imgs"+a[5]
                        # d["images_root"] = a
                        #
                        # b = d["points"].split("points")
                        # b = b[0] + "//points//" + b[2]
                        # d["points"] = b
                        #
                        # c = d["ego2world"].split("metas")
                        # c = c[0] + "//metas" + c[1]
                        # d["ego2world"] = c
                        #
                        # e = d["intrinsic"].split("metas")
                        # e = e[0] + "//metas" + e[1]
                        # d["intrinsic"] = e
                        #
                        # f = d["lidar_point"].split("lidar_point")
                        # f = f[0] + "//lidar_point" +f[1] + "lidar_point"+f[2]
                        # d["lidar_point"] = f

                        d_cleaned = {key.strip(): value.strip() for key, value in d.items()}
                        # a = d_cleaned["lidar2img"]
                        # import numpy as np
                        # a = np.load(a)
                        # one =  a[0]
                        # two = a[1]
                        # three = a[2]
                        # four = a[3]
                        # five = a[4]
                        # check(d_cleaned["lidar_point"])
                        dict_list.append(d_cleaned)
    data = {}
    data["metadata"] = {"version", "v1.0-urbanbis"}
    data["infos"] = dict_list
    if mode == "train":
        with open(train_pkl_file, 'wb') as pkl_out:
            print(len(data["infos"]))
            print(mode)
            pickle.dump(data, pkl_out)
    if mode == "val":
        with open(val_pkl_file, 'wb') as pkl_out:
            print(len(data["infos"]))
            print(mode)
            pickle.dump(data, pkl_out)
    if mode == "test":
        with open(test_pkl_file, 'wb') as pkl_out:
            print(len(data["infos"]))
            print(mode)
            pickle.dump(data, pkl_out)

# with open(test_pkl_file,'rb') as f:
#     d = pickle.load(f)['infos']
#     for i in d:
#         check(i["lidar_point"])
#         print("suc")
#     print(11)
# 写入数据到 txt 文件



# with open(val_pkl_file, 'wb') as pkl_out:
#     pickle.dump(data, pkl_out)
#
#     for item in dict_list:
#         print(item)

