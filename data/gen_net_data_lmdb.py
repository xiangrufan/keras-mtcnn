import sys
image_size = 48
sys.path.append('..\{0:d}net'.format(image_size))
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
import _pickle as pickle

anno_file = "wider_face_train.txt"
im_dir = r"WIDER_train\images"
pos_save_dir = r"{0:d}\positive".format(image_size)
part_save_dir = r"{0:d}\part".format(image_size)
neg_save_dir = r'{0:d}\negative'.format(image_size)
save_dir = r'.\{0:d}'.format(image_size)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_{0:d}.txt'.format(image_size)), 'w')
f2 = open(os.path.join(save_dir, 'neg_{0:d}.txt'.format(image_size)), 'w')
f3 = open(os.path.join(save_dir, 'part_{0:d}.txt'.format(image_size)), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print ("{:d} pics in total".format(num))
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
cls_list = []
roi_list = []


pos_cls_list = []
pos_roi_list = []
neg_cls_list = []
part_roi_list = []

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = list(map(float, annotation[1:]) )
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    idx += 1
    if idx % 100 == 0:
        print (idx, "images done")

    height, width, channel = img.shape


    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
        for i in range(6):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(int(-w * 0.2), int(w * 0.2))
            delta_y = npr.randint(int(-h * 0.2), int (h * 0.2))

            nx1 = max( int(x1 + w / 2 + delta_x - size / 2 ), 0)
            ny1 = max( int(y1 + h / 2 + delta_y - size / 2), 0)
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
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write(r"xx\positive\%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))

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
                # success = cv2.imwrite(save_file, resized_im)
                # if not success:
                #     raise Exception('Not writing file!')
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:   # output class label only
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write(r"xx\part\%s"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                im = resized_im
                h, w, ch = resized_im.shape
                if h != image_size or w != image_size:
                    im = cv2.resize(im, (image_size, image_size))
                im = np.swapaxes(im, 0, 2)
                # im -= 127
                im = (im - 127.5) / 127.5 #  it is wrong in original code
                label = -1
                roi = [float(offset_x1), float(offset_y1), float(offset_x2), float(offset_y2)]
                pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                part_roi_list.append([im, label, roi])

                # success = cv2.imwrite(save_file, resized_im)
                # if not success:
                #     raise Exception('Not writing file!')
                d_idx += 1
        box_idx += 1
        print ("{:d} images done, positive: {:d} part: {:d} negative: {:d}".format(idx, p_idx, d_idx, n_idx))



    neg_num = 0
    while neg_num < 10:
        size = npr.randint(40, int( min(width, height) / 2 ))
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.15:   # output class label only
            # Iou with all gts must below 0.3
            # save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(r"xx\negative\%s"%n_idx + ' 0\n')
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


part_keep = npr.choice(len(part_roi_list), size=p_idx * 1, replace=False)  # limit the amount of part sample to keep
neg_keep = npr.choice(len(neg_cls_list), size=p_idx*3, replace=False)   # limit the amount of negative sample to keep
for i in part_keep:
    roi_list.append(part_roi_list[i])
for i in neg_keep:
    cls_list.append(neg_cls_list[i])
cls_list.extend(pos_cls_list)
roi_list.extend(pos_roi_list)


fid = open("../{0:d}net/{0:d}/roi.imdb".format(image_size),'wb')
pickle.dump(roi_list, fid)
fid.close()

fid = open("../{0:d}net/{0:d}/cls.imdb".format(image_size),'wb')
pickle.dump(cls_list, fid)
fid.close()

f1.close()
f2.close()
f3.close()
