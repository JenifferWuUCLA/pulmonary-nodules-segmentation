# -*- coding:utf-8 -*-
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
import array
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 画3d图这行是必要的
import csv
import pandas as pd
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from scipy import ndimage as ndi

subset = "test_subset_all/"
# subset = "data_set/"
tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
# tianchi_path = "/home/jenifferwu/LUNA2016/"
# tianchi_subset_path = tianchi_path + subset

out_subset = "nerve-mine-2D/"
output_path = "/home/jenifferwu/IMAGE_MASKS_DATA/" + out_subset

coordinate_file = "imgs_mask_test_coordinate.csv"

###################################################################################

csvRows = []


def csv_row(seriesuid, coordX, coordY, coordZ, diameter_mm):
    new_row = []
    new_row.append(seriesuid)
    new_row.append(coordX)
    new_row.append(coordY)
    new_row.append(coordZ)
    new_row.append(diameter_mm)
    csvRows.append(new_row)


###################################################################################

class nodules_extract():
    def __init__(self, ):
        self.image_test = os.path.join(tianchi_path, subset)
        patients_test = [os.path.join(folder[0], f) for folder in os.walk(self.image_test) for f in
                         folder[2]]  # 获取所有病人三维图（包含mhd和raw文件）
        patients_test.sort()  # 排序，保证mhd和raw文件相邻
        self.mhds_test = patients_test[::2]  # 获取所有mhd格式图片
        self.raws_test = patients_test[1::2]  # 获取所有raw格式图片

    def readmhd(self, f):
        print("readmhd f: %s" % f)
        seriesuid = f.replace(os.path.join(tianchi_path, subset), "")
        seriesuid = seriesuid.replace(".mhd", "")
        print("readmhd seriesuid: %s" % seriesuid)
        itk_img = sitk.ReadImage(f)
        img_array = sitk.GetArrayFromImage(itk_img)
        spacing = np.array(itk_img.GetSpacing())
        origin = np.array(itk_img.GetOrigin())
        return img_array, spacing, origin

    def reshape(self, image, spacing, new_spacing=[1, 1, 1]):
        ''' 
        输入值：
            image: 原始三维图，shape的顺序是z,y,x
            spaceing:各个维度的切片厚度，对应的值需要是在，z,y,x的顺序
        返回值：
            image：重构后的三维图,默认对应1:1:1
            n_spacing：重构后的实际切片厚度，和输入值存在一定差异，因为实际做了四舍五入的处理。
        '''

        new_shape = np.round(image.shape * spacing / new_spacing)
        resize_factor = new_shape / image.shape

        image = ndi.interpolation.zoom(image, resize_factor, mode='nearest')
        n_spacing = spacing / resize_factor

        return image, n_spacing

    def get_segmented_lungs(self, im, threshold=-320):

        '''
        This funtion segments the lungs from the given 2D slice.
        Step 1: Convert into a binary image. 把肺部内外剂量大于threshold都置为0
        '''
        binary = im < threshold

        '''
        Step 2: Remove the blobs connected to the border of the image.
        '''
        cleared = clear_border(binary)

        '''
        Step 3: Label the image. 获取所有的联通区块标签
        '''
        label_image = label(cleared)

        '''
        Step 4: Keep the labels with 2 largest areas.
        '''
        regions = regionprops(label_image)  # 值为0的区域会被忽略
        keep_num = 2
        areas = [r.area for r in regions]
        areas.sort()
        if len(areas) > keep_num:
            for region in regions:
                if region.area < areas[-keep_num]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0

        '''
        Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
        '''
        selem = disk(2)  # 得到一个5*5的二维数组，正中心为1，中心周围2的单位也为1，其他为0
        binary = binary_erosion(binary, selem)  # 如果周围有空隙，腐蚀该点。缩小图像边界

        '''
        Step 6: Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall.
        '''
        selem = disk(10)
        binary = binary_closing(binary, selem)  # 先膨胀，在缩小 ,结果：连接图像内部断掉的部分。

        '''
        Step 7: Fill in the small holes inside the binary mask of lungs.
        '''
        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)

        '''
        Step 8: Superimpose the binary mask on the input image.
        '''
        get_high_vals = binary == 0
        im[get_high_vals] = -1024

        return binary, im

    def extract3d_lung(self, image):
        '''输入：原始3维 图
           输出：肺的3维图，以及肺的起始左边
           目的： 统一肺结节的坐标，因为肺结核概率和位置有一定有关             
        '''

        for piece in range(image.shape[0]):
            self.get_segmented_lungs(image[piece, :, :])

        couple_coordinate = [[np.min(i), np.max(i)] for i in
                             np.where(image > -1000)]  # (min_z,max_z),(min_y,max_y),(min_x,max_x)

        return image, couple_coordinate

    def get_lung_diamonds(self, image, coordinate, step=30, width=60, seriesuid=""):
        '''把分离出来的肺部，按照一个个立方体方式进行切割，其中边长为width，步长为step
        '''
        z, y, x = coordinate
        print("coordinate: ")
        print(coordinate)
        z_num = math.ceil((z[1] - z[0]) / step)
        y_num = math.ceil((y[1] - y[0]) / step)
        x_num = math.ceil((x[1] - x[0]) / step)
        print("z_num: %s" % z_num)
        print("y_num: %s" % y_num)
        print("x_num: %s" % x_num)

        diamonds = []
        marks = []
        # outputs = []
        for sz in range(int(z_num)):
            for sy in range(int(y_num)):
                for sx in range(int(x_num)):
                    print("sz: %d, sy: %d, sx: %d" % (sz, sy, sx))
                    s_z = z[0] + sz * step
                    s_y = y[0] + sy * step
                    s_x = x[0] + sx * step
                    diamonds.append(image[s_z:s_z + width, s_y:s_y + width, s_x:s_x + width])
                    # o = get_diamonds_output(nodes, s_x, s_y, s_z)
                    # outputs.append(o)
                    marks.append([s_z, s_y, s_x])
                    np.save(os.path.join(output_path,
                                         "data_images/test/images_{}_{}.npy".format(seriesuid, [s_z, s_y, s_x])),
                            image[s_z:s_z + width, s_y:s_y + width, s_x:s_x + width])
                    csv_row(seriesuid, s_x, s_y, s_z, "diameter_mm")
        return diamonds, marks

    def get_nodule(self):
        pass

    def exclude_noise(self, image):
        '''
                        把连通区块小于10的部分都标记被背景
                        把ct值小于-500部分都标记为背景
        '''
        bg = -1024
        image[image < -700] = bg

    def getsamples(self, f):
        print("getsamples f: %s" % f)
        seriesuid = f.replace(os.path.join(tianchi_path, subset), "")
        seriesuid = seriesuid.replace(".mhd", "")
        print("getsamples seriesuid: %s" % seriesuid)
        img_array, spacing, origin = self.readmhd(f)  # img_array顺序为z,y,x,spaceing顺序为x,y,z
        image, n_spacing = self.reshape(img_array, spacing[::-1])
        lungimage, couple_coordinate = self.extract3d_lung(image)
        lungimage = normalize(lungimage)
        self.get_lung_diamonds(lungimage, couple_coordinate, seriesuid=seriesuid)

    def getallsamples(self, mhds):
        for mhd in mhds:
            self.getsamples(mhd)

    def getxyz(self, mask):
        rst = np.where(mask != 0)
        x = rst[0]
        y = rst[1]
        z = rst[2]
        return x, y, z


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


PIXEL_MEAN = 0.25


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


if __name__ == '__main__':
    n = nodules_extract()
    csv_row("seriesuid", "coordX", "coordY", "coordZ", "diameter_mm")
    #     n.exclude_noise(np.load("../data/images/LKDS-00001_[150, 138, 89]_27612.0.npy"))
    test_data_path = os.path.join(tianchi_path, subset)
    test_images = glob(test_data_path + "*.mhd")
    for fcount, img_file in enumerate(tqdm(test_images)):
        print("fcount: %s" % str(fcount))
        n.getsamples(img_file)

    num_images = fcount + 1

    # Write out the imgs_mask_test_coordinate CSV file.
    print(os.path.join(output_path + "data_images/test/csv/", coordinate_file))
    csvFileObj = open(os.path.join(output_path + "data_images/test/csv/", coordinate_file), 'w')
    csvWriter = csv.writer(csvFileObj)
    for row in csvRows:
        # print row
        csvWriter.writerow(row)
    csvFileObj.close()
