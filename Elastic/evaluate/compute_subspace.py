import pandas as pd
from PIL import Image
import numpy as np
import os
import torch
from sewar.full_ref import vifp

if __name__ == '__main__':
    data = pd.read_csv(r'I:\Elastic Editing\Elastics\results\result_stylegan2_offcial\25 160.csv')
    print(data.mean(), data.var())

    # load raw images and updated images
    # raw_dir = r'I:\Elastic Editing\Elastics\results\result_stylegan2_offcial\raw'
    # raw_dir = './results/result_stylegan2_offcial/raw'
    # raw_images = [os.path.join(raw_dir, x) for x in os.listdir(raw_dir)]
    # raw_images = sorted(raw_images)
    # raw_images = list(filter(os.path.isfile, raw_images))
    #
    # updated_dir = './results/result_stylegan2_offcial/update'
    # updated_images = [os.path.join(updated_dir, x) for x in os.listdir(updated_dir)]
    # updated_images = sorted(updated_images)
    # updated_images = list(filter(os.path.isfile, updated_images))
    #
    # # when VIF metric is not correct, we compute it again with generated images
    # vif_list = list()
    # for images_index in range(len(raw_images)):
    #     im = np.array(Image.open(raw_images[images_index]))
    #     im1 = np.array(Image.open(updated_images[images_index]))
    #     vif = vifp(im, im1)
    #     vif_list.append(vif)
    #
    # np.save("vif.npy", np.array(vif_list))
    #
    # print(np.mean(vif_list), np.var(vif_list))

    print()
