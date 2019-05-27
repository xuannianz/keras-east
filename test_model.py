from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.xception import preprocess_input
from keras.utils.vis_utils import plot_model

model = ResNet50(include_top=False, input_shape=(512, 512, 3))
plot_model(model, show_shapes=True, to_file='resnet50.jpg')
