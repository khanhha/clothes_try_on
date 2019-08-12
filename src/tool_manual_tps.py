import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand
import cv2 as cv
from src.moving_least_square import mls_rigid_deformation_inv, mls_similarity_deformation_inv, mls_affine_deformation_inv

def estimate_tps_transform(sources, targets):
    assert sources.shape[0] == targets.shape[0]
    N = sources.shape[0]
    matches = list()
    for i in range(N):
        matches.append(cv.DMatch(i, i, 0))
    sources = np.array(sources).reshape((1, -1, 2)).astype(np.float32)
    targets = np.array(targets).reshape((1, -1, 2)).astype(np.float32)

    tps = cv.createThinPlateSplineShapeTransformer(regularizationParameter = 10)
    tps.estimateTransformation(sources, targets, matches)
    retval, test_pnts = tps.applyTransformation(sources)
    print("mean error = ", np.mean(test_pnts - targets))
    return tps

def tps_warp_image(clothes_img, clothes_pnt, hm_points):
    tps = estimate_tps_transform(hm_points, clothes_pnt)
    img_warp = clothes_img.copy()
    tps.warpImage(clothes_img, img_warp, flags=cv.INTER_CUBIC)
    return img_warp

def warp_image(img, sources, targets, method):
    if method == 'tps':
        tps = estimate_tps_transform(sources, targets)
        img_warp = img.copy()
        tps.warpImage(img, img_warp, flags=cv.INTER_CUBIC)
        return img_warp
    elif method == 'rigid_mls':
        img_warp = mls_rigid_deformation_inv(img, targets, sources)
        return img_warp
    else:
        assert False, 'non support method'

def pick_image(clothes_path, human_path):
    def draw(fig, axes, img0, img1, left_points, right_points, colors):
        axes[0].clear()
        axes[1].clear()
        axes[0].imshow(img0)
        axes[1].imshow(img1)
        #all_colors = [k for k, v in pltc.cnames.items()]
        for i, pnt in enumerate(left_points):
            print(pnt)
            axes[0].scatter(pnt[0],pnt[1], c=np.array(colors[i]).reshape(-1,3))

        for i, pnt in enumerate(right_points):
            axes[1].scatter(pnt[0],pnt[1], c=np.array(colors[i]).reshape(-1,3))

        fig.canvas.draw()

    # picking images (matplotlib.image.AxesImage)
    fig, axes = plt.subplots(2,2, gridspec_kw = {'wspace':0, 'hspace':0})
    axes = axes.flatten()
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

    #img0 = (np.random.rand(255,255)*255).astype(np.uint8)
    #img1 = (np.random.rand(255,255)*255).astype(np.uint8)
    img0 = cv.imread(clothes_path)
    img1 = cv.imread(human_path)
    img0 = img0[:,:,::-1]
    img1 = img1[:,:,::-1]
    left_pnts = []
    right_pnts = []
    colors = []

    #warp_method = mls_rigid_deformation_inv
    #warp_method = mls_similarity_deformation_inv
    warp_methods =      [tps_warp_image, mls_affine_deformation_inv, mls_similarity_deformation_inv, mls_rigid_deformation_inv]
    warp_method_names = ['tps_warp_image', 'mls_affine_deformation_inv', 'mls_similarity_deformation_inv', 'mls_rigid_deformation_inv']
    warp_method_idx = 0
    fig.suptitle(warp_method_names[warp_method_idx])
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img0)
    axes[3].imshow(img1)
    def on_key_press_event(ev):
        nonlocal warp_method_idx
        if ev.key == 'a':
            n0 = len(left_pnts)
            n1 = len(right_pnts)
            if n0 == n1 and n0 >= 3:
                #img2 = warp_image(img0, np.array(right_pnts), np.array(left_pnts), 'rigid_mls')
                img2 = warp_methods[warp_method_idx](img0, np.array(left_pnts), np.array(right_pnts))
                axes[2].imshow(img2)
                axes[3].imshow(img1)
                axes[3].imshow(img2, alpha=0.5)
                fig.canvas.draw()
        elif ev.key == 'd':
            warp_method_idx += 1
            warp_method_idx = warp_method_idx % len(warp_methods)
            fig.suptitle(warp_method_names[warp_method_idx])
            fig.canvas.draw()
        elif ev.key == 'c':
            right_pnts.clear()
            left_pnts.clear()
            colors.clear()
            draw(fig, axes, img0, img1, left_pnts, right_pnts, colors)

    def on_button_press_event(ev):
        pnt = (int(ev.xdata), int(ev.ydata))
        n0 = len(left_pnts)
        n1 = len(right_pnts)
        if ev.inaxes == axes[0] and (n0 == n1 or n0 == n1-1):
            left_pnts.append(pnt)
        elif ev.inaxes == axes[1] and (n0 == n1 or n1 == n0-1):
            right_pnts.append(pnt)

        nmax = max(len(left_pnts), len(right_pnts))
        if nmax > len(colors):
            c = np.random.random(3)
            colors.append(c)

        draw(fig, axes, img0, img1, left_pnts, right_pnts, colors)

    #fig.canvas.mpl_connect('pick_event', onpick4)
    fig.canvas.mpl_connect('button_press_event', on_button_press_event)
    fig.canvas.mpl_connect('key_press_event', on_key_press_event)

if __name__ == '__main__':
    #pick_simple()
    #pick_custom_hit()
    #pick_scatter_plot()
    #plt.scatter(0, 1, c=np.random.random(3))
    #plt.show()
    #dir_c = '/home/khanhhh/data_1/projects/thesis/dataset_fashion/viton_resize/test/cloth/'
    #dir_i = '/home/khanhhh/data_1/projects/thesis/dataset_fashion/viton_resize/test/image/'
    dir_c = '/home/khanhhh/data_1/projects/thesis/dataset_fashion/viton_resize/GMM_train_result/gmm_final.pth/train/cloth/'
    dir_i = '/home/khanhhh/data_1/projects/thesis/dataset_fashion/viton_resize/GMM_train_result/gmm_final.pth/train/image/'
    #id = '000109'
    id = '000040'
    clothes_path = f'{dir_c}/{id}_1.jpg'
    human_path = f'{dir_i}{id}_0.jpg'
    pick_image(clothes_path, human_path)
    plt.show()