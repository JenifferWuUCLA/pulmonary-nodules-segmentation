from glob import glob
import shutil
import os


# subset = "train_dataset/"
subset = "data_set/"
# tianchi_path = "/media/ucla/32CC72BACC727845/tianchi/"
tianchi_path = "/home/jenifferwu/LUNA2016/"
tianchi_subset_path = tianchi_path + subset
train_path = tianchi_subset_path + "train/"
test_path = tianchi_subset_path + "test/"


file_list = glob(tianchi_subset_path + "*.mhd")
num_images = len(file_list)
test_i = int(0.2 * num_images)


index = 0
for img_file in file_list:
    if index < test_i:
        print("test image: %s" % img_file)
        shutil.copy(img_file, test_path)
        filename = img_file.replace(tianchi_subset_path, "")
        raw_file = filename.replace(".mhd", "") + ".raw"
        print("test raw: %s" % os.path.join(tianchi_subset_path, raw_file))
        shutil.copy(os.path.join(tianchi_subset_path, raw_file), test_path)
        index += 1

    if index >= test_i:
        print("train image: %s" % img_file)
        shutil.copy(img_file, train_path)
        filename = img_file.replace(tianchi_subset_path, "")
        raw_file = filename.replace(".mhd", "") + ".raw"
        print("train raw: %s" % os.path.join(tianchi_subset_path, raw_file))
        shutil.copy(os.path.join(tianchi_subset_path, raw_file), train_path)
        index += 1