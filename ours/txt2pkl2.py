
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

dict_list = []
map_name= "yuehai"
file_path =  root_path+map_name+"//"+"2024-11-14-10-39-41//datas.txt"
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

            d_cleaned = {key.strip(): value.strip() for key, value in d.items()}
            dict_list.append(d_cleaned)
data = {}
data["metadata"] = {"version", "v1.0-urbanbis"}
data["infos"] = dict_list
with open(test_pkl_file, 'wb') as pkl_out:
    print(len(data["infos"]))
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

