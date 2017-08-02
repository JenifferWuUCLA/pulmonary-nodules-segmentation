import numpy as np
import sys
import caffe
import os

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support


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
# Switching to GPU mode
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

model_def = caffe_root + 'models/pulmonary_nodules_net_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/pulmonary_nodules_net_caffenet/caffenet_train_iter_30000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean of PulmonaryNodules images (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/pulmonary_nodules_net/pulmonary_nodules_net_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# 3. GPU classification
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

# set the tested file and images path
TEST_FILE = caffe_root + 'data/pulmonary_nodules/val.txt'
inputs = []
groundTruths = []
predictions = []
file = open(TEST_FILE)
for each_line in file:
    (i, g) = each_line.split(' ', 1)
    inputs.append(i)
    groundTruths.append(int(g))

for idx in range(len(inputs)):
        image = caffe.io.load_image(caffe_root + 'data/pulmonary_nodules/val/' + inputs[idx])
        transformed_image = transformer.preprocess('data', image)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image
        # perform classification
        output = net.forward()
        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        predictions.append(output_prob.argmax())

target_names = ['It is not a real lung nodule.', 'It is a real lung nodule.']
print(classification_report(groundTruths, predictions, target_names=target_names))

# estimate precision and recall of the Caffe model
precision_score(groundTruths, predictions, average='macro') 
precision_score(groundTruths, predictions, average='micro') 
precision_score(groundTruths, predictions, average='weighted') 
precision_score(groundTruths, predictions, average=None) 

recall_score(groundTruths, predictions, average='macro')  
recall_score(groundTruths, predictions, average='micro')  
recall_score(groundTruths, predictions, average='weighted')  
recall_score(groundTruths, predictions, average=None)  

precision_recall_fscore_support(groundTruths, predictions, average='macro')
precision_recall_fscore_support(groundTruths, predictions, average='micro')
precision_recall_fscore_support(groundTruths, predictions, average='weighted')