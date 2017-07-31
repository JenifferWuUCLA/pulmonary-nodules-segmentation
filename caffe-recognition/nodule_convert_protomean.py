import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('data/pulmonary_nodules/pulmonary_nodules_net_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save('python/caffe/pulmonary_nodules/pulmonary_nodules_net_mean.npy', out)
