import caffe
import numpy as np


caffe_root = '/root/code/caffe/'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(caffe_root + 'data/pulmonary_nodules/pulmonary_nodules_net_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save(caffe_root + 'python/caffe/pulmonary_nodules_net/pulmonary_nodules_net_mean.npy', out)
