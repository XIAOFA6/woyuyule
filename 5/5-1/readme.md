### 预备工作
（1） 首先安装程序运行必备的一些库。
```
!pip install tflite-model-maker
```
（2）接下来，导入相关的库
```
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
```

### 模型训练
（1） 获取数据：本实验先从较小的数据集开始训练，当然越多的数据，模型精度更高。
```
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```
这里从storage.googleapis.com中下载了本实验所需要的数据集。image_path可以定制，默认是在用户目录的.keras\datasets中。

（2）运行示例：一共需4步完成。
1:加载数据集，并将数据集分为训练数据和测试数据。
```
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```
```
INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
2023-06-07 01:26:30.054433: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/envs/tf/lib/python3.8/site-packages/cv2/../../lib64:
2023-06-07 01:26:30.054473: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-06-07 01:26:30.054499: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-9ee10c): /proc/driver/nvidia/version does not exist
2023-06-07 01:26:30.069245: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```
2:训练Tensorflow模型
```
model = image_classifier.create(train_data)
```

```
INFO:tensorflow:Retraining the models...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
 rasLayerV1V2)                                                   
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 3,419,429
Trainable params: 6,405
Non-trainable params: 3,413,024
_________________________________________________________________
None
Epoch 1/5
2023-06-07 01:26:51.325810: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 19267584 exceeds 10% of free system memory.
2023-06-07 01:26:51.381185: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 19267584 exceeds 10% of free system memory.
2023-06-07 01:26:51.394797: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2023-06-07 01:26:51.452954: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 19267584 exceeds 10% of free system memory.
2023-06-07 01:26:51.506970: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 19267584 exceeds 10% of free system memory.
103/103 [==============================] - 58s 535ms/step - loss: 0.8652 - accuracy: 0.7715
Epoch 2/5
103/103 [==============================] - 57s 548ms/step - loss: 0.6520 - accuracy: 0.8981
Epoch 3/5
103/103 [==============================] - 54s 518ms/step - loss: 0.6236 - accuracy: 0.9084
Epoch 4/5
103/103 [==============================] - 53s 516ms/step - loss: 0.6013 - accuracy: 0.9266
Epoch 5/5
103/103 [==============================] - 54s 524ms/step - loss: 0.5867 - accuracy: 0.9375
```

3:评估模型
```
loss, accuracy = model.evaluate(test_data)
```

```
12/12 [==============================] - 7s 436ms/step - loss: 0.6347 - accuracy: 0.8992
```

4:导出Tensorflow Lite模型
```
model.export(export_dir='.')
```
2023-06-07 01:32:01.038924: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
INFO:tensorflow:Assets written to: /tmp/tmpanp4kif3/assets
INFO:tensorflow:Assets written to: /tmp/tmpanp4kif3/assets
2023-06-07 01:32:05.523797: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-06-07 01:32:05.524011: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-06-07 01:32:05.570800: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
  function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 25.432ms.
  function_optimizer: function_optimizer did nothing. time = 0.017ms.

/opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "
2023-06-07 01:32:06.812803: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
2023-06-07 01:32:06.812856: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
INFO:tensorflow:Label file is inside the TFLite model with metadata.
fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
INFO:tensorflow:Label file is inside the TFLite model with metadata.
INFO:tensorflow:Saving labels in /tmp/tmpp2xdsi0r/labels.txt
INFO:tensorflow:Saving labels in /tmp/tmpp2xdsi0r/labels.txt
INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
```
这里导出的Tensorflow Lite模型包含了元数据(metadata),其能够提供标准的模型描述。导出的模型存放在Jupyter Notebook当前的工作目录中。


