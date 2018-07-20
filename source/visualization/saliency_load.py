from vis.utils import utils
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

# img1 = utils.load_img('images/ouzel1.jpg', target_size=(224, 224))
# img2 = utils.load_img('images/ouzel2.jpg', target_size=(224, 224))

img1 = utils.load_img('images/ouzel1.jpg')
img2 = utils.load_img('images/ouzel2.jpg')


f, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(img2)