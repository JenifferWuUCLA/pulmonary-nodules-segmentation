import numpy as np
import sys
import caffe
import os
from glob import glob
import csv


subset = "val_subset_all/"
output_path = "/home/ucla/Downloads/tianchi/" + subset

probability_file = "instant-recognition/pulmonary_nodule_probability.csv"

csvRows = []

def csv_row(image_name, probability, label):
    new_row = []
    new_row.append(image_name)
    new_row.append(probability)
    new_row.append(label)
    csvRows.append(new_row)

# 1. Setup
# set up Python environment: numpy for numerical routines
# The caffe module needs to be on the Python path; we'll add it here explicitly.
caffe_root = '/root/code/caffe/'
sys.path.insert(0, caffe_root + 'python')

# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
if os.path.isfile(caffe_root + 'models/pulmonary_nodules_net_caffenet/caffenet_train_iter_30000.caffemodel'):
    print('CaffeNet found.')
else:
    print('CaffeNet not found.')

# 2. Load net and set up input preprocessing
caffe.set_mode_cpu()

model_def = caffe_root + 'models/pulmonary_nodules_net_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/pulmonary_nodules_net_caffenet/caffenet_train_iter_30000.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# load the mean PulmonaryNodules image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# 3. GPU classification
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

test_images = glob(os.path.join(output_path, "images_*.jpg"))
for test_image in test_images:
    image = caffe.io.load_image(test_image)
    transformed_image = transformer.preprocess('data', image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    # Switching to GPU mode
    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    # perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print('predicted class is:', output_prob.argmax())

    # load PulmonaryNodulesNet labels
    labels_file = caffe_root + 'data/pulmonary_nodules/synset_words.txt'
    # if not os.path.exists(labels_file):
    #     !../data/pulmonary_nodules/get_pulmonary_nodules.sh

    labels = np.loadtxt(labels_file, str, delimiter='\t')

    print('output label:', labels[output_prob.argmax()])

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

    print('probabilities and labels:')
    z_list = zip(output_prob[top_inds], labels[top_inds])
    print(z_list)
    for index in range(len(z_list)):
        z_item = z_list[index]
        probability, label = z_item[0], z_item[1]
        csv_row(test_image, probability, label)


# 6. Try your own image start
# transform it and copy it into the net
test_image = "images_0000_0000_0.jpg "
image = caffe.io.load_image(output_path + test_image)
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# Switching to GPU mode
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

labels = np.loadtxt(labels_file, str, delimiter='\t')
print('output label:', labels[output_prob.argmax()])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]

print('probabilities and labels:')
z_list = zip(output_prob[top_inds], labels[top_inds])
print(z_list)
for index in range(len(z_list)):
    z_item = z_list[index]
    probability, label = z_item[0], z_item[1]
    csv_row(test_image, probability, label)
# 6. Try your own image end


# Write out the pulmonary_nodule_probability CSV file.
print(os.path.join(output_path, probability_file))
csvFileObj = open(os.path.join(output_path, probability_file), 'w')
csvWriter = csv.writer(csvFileObj)
for row in csvRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()