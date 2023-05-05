import wget
model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
wget.download(model_link)
import tarfile
tar = tarfile.open('/content/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz')
tar.extractall('.') 
tar.close()
