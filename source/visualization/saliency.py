from keras.applications import VGG16
from vis.utils import utils
from keras import activations

model = VGG16(weights='imagenet', include_top=True)

layer_idx = utils.find_layer_idx(model, 'predictions')

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

