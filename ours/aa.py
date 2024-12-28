import pickle

import numpy as np

with open("/mnt/d/project/SurroundOcc-main/data/val.pkl", 'rb') as file:
    data = pickle.load(file)
    data = data ["infos"]
    print(len(data))
    a = data[0]

a = np.load("/mnt/d/project/SurroundOcc-main/visual_dir/102/pred.npy")
# n_data = np.load("/media/vcc/LinuxData/project/urbanbisFly/photos/scene/points/one/0.npy")
print(1)