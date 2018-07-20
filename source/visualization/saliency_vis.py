from keras.applications import VGG16
from vis.utils import utils
from keras import activations
from matplotlib import pyplot as plt
#
from vis.visualization import visualize_saliency, overlay

#
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam

model = VGG16(weights='imagenet', include_top=True)

layer_idx = utils.find_layer_idx(model, 'predictions')

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

plt.rcParams['figure.figsize'] = (18, 6)

img1 = utils.load_img('images/ouzel1.jpg', target_size=(224, 224))
img2 = utils.load_img('images/ouzel2.jpg', target_size=(224, 224))

f, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(img2)

layer_idx = utils.find_layer_idx(model, 'predictions')

#
f, ax = plt.subplot(1,2)
for i, img in enumerate([img1, img2]):
    grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img)

    ax[i].imshow(grads, cmap='jet')

#
for modifier in ['guided', 'relu']:
    plt.figure()
    f, ax = plt.subplot(1,2)
    plt.suptitle(modifier)
    for i, img in enumerate([img1, img2]):
        grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img, backprop_modifier=modifier)
        ax[i].imshow(grads, cmap='jet')


#
for modifier in [None, 'guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model, layer_idx, filter_indices=20,
                              seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original images.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[i].imshow(overlay(jet_heatmap, img))
