#-*- coding:utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

def scale_data(data, data_min = -1, data_max = 1):
    data = np.array(data)
    print("in sale data when ploting: data min:", np.min(data), "data max:", np.max(data), "data mean:", np.mean(data))
    # data = (data - np.min(data)) / (np.abs(np.max(data) - np.min(data)) +0.001) * 255 # .0001 supports for all zero input
    data = (data - data_min) / np.abs(data_max - data_min) * 255 # .0001 supports for all zero input
    data = np.clip(data, 0, 255)
    data = data.astype(np.uint8)
    return data

def make_video(data, path):
    # expected shape of data: 155 192 192
    frames = scale_data(data)
    f,w,h = frames.shape
    frames[0][0][0] = 0
    frames[-1][-1][-1] = 255

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path + ".avi", fourcc, 20.0, (w, h),isColor=False)

    for frame in frames:
        out.write(frame)

    # Release everything when job is finished
    out.release()

def make_plot(data, path, n_row=4):
    # expected shape of data: 152 192 192
    d,w,h = data.shape
    print("saving plot in: ",path)
    data = scale_data(data)
    data[0][0][0] = 0
    data[-1][-1][-1] = 255
    
    canvas = np.zeros((w*n_row, h*n_row))
    for i in range(n_row):
        for j in range(n_row):
            current_plot = n_row*i+j+1
            canvas[i*w:(i+1)*w,j*h:(j+1)*h] = data[d//(n_row**2)*current_plot, :, :]
    
    # use a larger plot window
    plt.figure(figsize=(20, 20))
    plt.imshow(canvas, cmap="gray")
    plt.savefig(path + ".png")
    plt.close()

    max_size = np.max(data.shape)

    canvas_3_view = np.zeros((max_size,max_size*3))
    canvas_3_view[:w,:h] = data[d//2,:,:]
    canvas_3_view[:d,max_size:max_size+h] = data[:,w//2,:]
    canvas_3_view[:d,max_size*2:max_size*2+w] = data[:,:,h//2]

    plt.figure(figsize=(20, 20))
    plt.imshow(canvas_3_view, cmap="gray")
    plt.savefig(path + "_3view.png")
    plt.close()


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (d,w,h). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (d,w,h).
    """

    d, w, h = img_shape

    mask = np.zeros((d, w, h), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5]] = 1

    return mask

def random_bbox(img_shape=(192,192,152), max_bbox_shape=(128, 128, 64), max_bbox_delta=40, min_margin=20):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)
        
    img_h, img_w, img_d = img_shape[:3]
    max_mask_h, max_mask_w, max_mask_d = max_bbox_shape
    max_delta_h, max_delta_w, max_delta_d = max_bbox_delta
    margin_h, margin_w, margin_d = min_margin

    if max_mask_h > img_h or max_mask_w > img_w or margin_d > img_d:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w
            or max_delta_d // 2 * 2 >= max_mask_d):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w or img_d - max_mask_d < 2 * margin_d:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    max_d = img_d - margin_d - max_mask_d
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    d = np.random.randint(margin_d, max_d)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    delta_d = np.random.randint(0, max_delta_d // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    pos_d = d + delta_d

    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    d = max_mask_d - delta_d
    return (top, left, pos_d , h, w, d)


if __name__ == '__main__':
    data = np.ones((100, 128, 128)) * 1
    data[10:-10,10:-10,10:-10] = 0
    make_plot(data, "test")
    make_video(data, "test")