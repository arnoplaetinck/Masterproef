'''
import GPUtil

GPUtil.showUtilization()
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
print(deviceIDs)
deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
print(deviceIDs)
'''

import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())
# my output was => ['/device:CPU:0']
# good output must be => ['/device:CPU:0', '/device:GPU:0']

name = tf.test.gpu_device_name()
print(name)
if tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None):
    print("true")
else:
    print("false")
