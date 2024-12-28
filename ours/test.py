# 写入数据到 txt 文件，如果文件不存在则创建它
txt_file = 'example.txt'

import pickle
pkl = "/mnt/d/project/SurroundOcc-main/data/nuscenes_infos_val.pkl"
test_pkl = "/mnt/d/project/SurroundOcc-main/data/nuscenes_infos_test.pkl"
with open(pkl,'rb') as f:
    d = pickle.load(f)
data = d['infos'][:3]
meta = d['metadata']
datas  =  {}
datas["metadata"] = meta
datas["infos"] = data
with open(test_pkl, 'wb') as pkl_out:
    pickle.dump(datas, pkl_out)
print(11)

