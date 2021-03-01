#/bin/bash

version=$1


wget -O noisy_student_efficientnet-b${version}.tar.gz https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b${version}.tar.gz
tar -xvf noisy_student_efficientnet-b$version.tar.gz
wget -O efficientnet_weight_update_util.py https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py
python efficientnet_weight_update_util.py --model b$version --notop --ckpt noisy-student-efficientnet-b$version --o efficientnetb${version}_notop.h5