import sys
image_size = 48
sys.path.append('..\{0:d}net'.format(image_size))
import numpy as np
import cv2
import os
import numpy.random as npr
from data.utils import IoU
import _pickle as pickle
import matplotlib.pyplot as plt

anno_file = r"C:\Users\xiangru\Desktop\Dataku\celebA\Anno\list_bbox_celeba.txt"
landmark_file = r"C:\Users\xiangru\Desktop\Dataku\celebA\Anno\list_landmarks_celeba.txt"
im_dir = r"C:\Users\xiangru\Desktop\Dataku\celebA\Img\img_celeba"

with open(anno_file, 'r') as f:
    annotations = f.readlines()
with open(landmark_file, 'r') as f2:
    landmark_positions_list = f2.readlines()
num = len(annotations)
print ("{:d} pics in total".format(num))
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
pts_idx = 0
cls_list = []
roi_list = []


pos_cls_list = []
pos_roi_list = []
neg_cls_list = []
part_roi_list = []
pts_list = []
for id_annos in range (2,10000):  # too large, use only first 10000 data
    annotation = annotations[id_annos]
    # annotation = annotation.strip().split(' ')
    annotation = annotation.strip().split()
    im_path = annotation[0]
    bbox = list(map(float, annotation[1:]) )
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    landmark_positions = landmark_positions_list[id_annos]
    landmark_positions = landmark_positions.strip().split()
    pts_list_raw = list(map(float, landmark_positions[1:]))   # need to convert to relative pts positions
    # print aa

    file_basename = os.path.splitext(im_path)[0]
    img = cv2.imread(os.path.join(im_dir, file_basename + '.jpg'))  # the celebA database seems to have used wrong name
    idx += 1
    if idx % 100 == 0:
        print (idx, "images done")

    height, width, channel = img.shape


    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, w, h = box
        w_box = w
        h_box = h
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        x2 = w + x1 -1
        y2 = h + y1 -1
        box = np.array([x1,y1,x2,y2])

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
        for i in range(20):
            # size = npr.randint(int(min(w, h) * 0.9), np.ceil(1.25 * max(w, h)))# enlarge size here
            #
            # # delta here is the offset of box center
            # delta_x = npr.randint(int(-w * 0.1), int(w * 0.1))   # reduce uncertainty here
            # delta_y = npr.randint(int(-h * 0.1), int (h * 0.1))
            # nx1 = max( int(x1 + w / 2 + delta_x - size / 2 ), 0)   # position of selected box part
            # ny1 = max( int(y1 + h / 2 + delta_y - size / 2), 0)

            size = int(min(w,h))  # enlarge size here
            # delta here is the offset of box center

            nx1 = max(int (x1), 0)  # position of selected box part
            ny1 = max(int (y1), 0)


            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)

            if IoU(crop_box, box_) >= 0.65:   # output both position vector and class label
                im = resized_im
                h, w, ch = resized_im.shape
                if h != image_size or w != image_size:
                    im = cv2.resize(im, (image_size, image_size))
                im = np.swapaxes(im, 0, 2)
                im = (im - 127.5) / 127.5
                label = 1
                roi = [-1, -1, -1, -1]
                pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                pos_cls_list.append([im, label, roi])
                roi = [float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)]
                pos_roi_list.append([im, label, roi])
                p_idx += 1

                (px1,py1,px2,py2,px3,py3,px4,py4,px5,py5) = pts_list_raw
                if min ([px1, px2, px3,px4, px5]) > nx1 and min ([py1, py2, py3,py4, py5]) > ny1 \
                        and max ([px1, px2, px3,px4, px5]) < nx2 and max([py1, py2, py3,py4, py5]) < ny2 :
                    # pts_list.append([im, label, roi, pts])
                    pts_xs = pts_list_raw[0::2]
                    pts_ys = pts_list_raw[1::2]
                    # pts_xs_real =(pts_xs +1 -x1 )/w   # not planning to use numpy here, so list comprehension here
                    # pts_xs_real = [(tmpval -x1 )/w_box for tmpval in pts_xs]
                    # # pts_ys_real = (pts_ys + 1 - y1) / h
                    # pts_ys_real = [(tmpval - y1) / h_box for tmpval in pts_ys]
                    pts_xs_real = [(tmpval - x1 )/size for tmpval in pts_xs]# mistake here last time
                    pts_ys_real = [(tmpval - y1 )/size for tmpval in pts_ys]

                    pts = [0]*10
                    pts[0::2] = pts_xs_real  # should be numbers between 0 and 1 .
                    pts[1::2] = pts_ys_real

                    # label = -1
                    # roi = [-1, -1, -1, -1]
                    label = 1
                    roi = [float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)]
                    pts_list.append([im, label, roi, pts])
                    pts_idx += 1

            elif IoU(crop_box, box_) >= 0.4:   # output class label only

                im = resized_im
                h, w, ch = resized_im.shape
                if h != image_size or w != image_size:
                    im = cv2.resize(im, (image_size, image_size))
                im = np.swapaxes(im, 0, 2)
                im = (im - 127.5) / 127.5
                label = -1
                roi = [float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)]
                pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                part_roi_list.append([im, label, roi])
                d_idx += 1
        box_idx += 1
        print ("{:d} images done, Lanmark{:d} positive:  {:d} part: {:d} negative: {:d}".format(idx, pts_idx,p_idx, d_idx, n_idx))



    neg_num = 0
    while neg_num < 10:
        if int( min(width, height) / 2 )>40:
            size = npr.randint(40, int( min(width, height) / 2 ))
        else:
            size = npr.randint(int(min(width, height) / 2), 40)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:   # output class label only
            # Iou with all gts must below 0.3
            im = resized_im
            h, w, ch = resized_im.shape
            if h != image_size or w != image_size:
                im = cv2.resize(im, (image_size, image_size))
            im = np.swapaxes(im, 0, 2)
            im = (im - 127.5) / 127.5
            label = 0
            roi = [-1, -1, -1, -1]
            pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            neg_cls_list.append([im, label, roi])
            n_idx += 1
            neg_num += 1
            # success = cv2.imwrite(save_file, resized_im)
            # if not success:
            #     raise Exception('Not writing file!')


if p_idx < len(part_roi_list):
    part_keep = npr.choice(len(part_roi_list), size=p_idx * 1, replace=False)  # limit the amount of part sample to keep
    for i in part_keep:
        roi_list.append(part_roi_list[i])
else:
    roi_list.extend(part_roi_list)

if p_idx < len(neg_cls_list):
    neg_keep = npr.choice(len(neg_cls_list), size=p_idx*3, replace=False)   # limit the amount of negative sample to keep
    for i in neg_keep:
        cls_list.append(neg_cls_list[i])
else:
    cls_list.extend(neg_cls_list)


cls_list.extend(pos_cls_list)
roi_list.extend(pos_roi_list)

# before dumping data, check shape of data
for iline, line in enumerate(roi_list):
    if len(line) != 3:
        raise Exception('roi_list shape is wrong')
for iline, line in enumerate(pts_list):
    if len(line) != 4:
        raise Exception('pts_list shape is wrong')
for iline, line in enumerate(cls_list):
    if len(line) != 3:
        raise Exception('cls_list shape is wrong')

fid = open("./{0:d}net/{0:d}/roi.imdb".format(image_size),'wb')
pickle.dump(roi_list, fid)
fid.close()
fid = open("./{0:d}net/{0:d}/pts.imdb".format(image_size),'wb')
pickle.dump(pts_list, fid)
fid.close()
fid = open("./{0:d}net/{0:d}/cls.imdb".format(image_size),'wb')
pickle.dump(cls_list, fid)
fid.close()

