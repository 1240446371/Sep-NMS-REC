import os.path as osp
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# image£¬box and token and score
def draw_attention(data, loader, n):
    image_root = '/COCO/'

    plt.figure()
    ax = plt.gca()
    I = io.imread(osp.join(image_root, data['images']['file_name']))
    ax.imshow(I)

    bbox = loader.Anns[data['ref'][n]['ann_id']]['box']
    box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='#1f953b', linewidth=2)
    ax.add_patch(box_plot)

    plt.axis('off')
    plt.show()
    plt.cla()
    plt.close("all")