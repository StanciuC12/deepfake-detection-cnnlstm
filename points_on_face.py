import os
import numpy as np
import time
import pandas as pd
import skimage
import cv2


def crop_square(img, face_points, final_dim=(299, 299)):

    minx, maxx, miny, maxy = int(np.min(face_points[:, 1])), int(np.max(face_points[:, 1])), int(np.min(face_points[:, 0])), int(np.max(face_points[:, 0]))
    sides = [maxy - miny, maxx - minx]
    cropped_img = img[minx:maxx, miny:maxy]

    top, bottom, left, right = 0, 0, 0, 0
    make_border = True
    if sides[0] > sides[1]:
        diff = sides[0] - sides[1]
        top = int(diff / 2)
        bottom = int(diff / 2) if diff % 2 == 0 else int(diff / 2) + 1
    elif sides[1] > sides[0]:
        diff = sides[1] - sides[0]
        right = int(diff / 2)
        left = int(diff / 2) if diff % 2 == 0 else int(diff / 2) + 1
    else:
        make_border = False

    if make_border:
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        new_im = cv2.resize(new_im, final_dim)
    else:
        new_im = cv2.resize(cropped_img, final_dim)

    return new_im


if __name__ == "__main__":

    ds_adr = r'C:\Users\user\Desktop\ML\AI4Media\Datasets\DFDC'
    csv_path = r'E:\processed_DFDC'
    save_path = r'E:\saved_img_dfdc'

    x_cols = [' x_' + str(x) for x in range(68)]
    y_cols = [' y_' + str(x) for x in range(68)]

    try:
        pts_df = pd.read_csv(r'E:\processed_celebdf\id27_0006.csv')
        poz_cols = [x for x in pts_df.columns if x.startswith(' x') or x.startswith(' y')]
        pts_df.loc[:, poz_cols] = pts_df.loc[:, poz_cols].astype(int)
    except:
        pass

    cap = cv2.VideoCapture(r'C:\Users\user\Desktop\ML\AI4Media\Datasets\CelebDF\Celeb-real\id27_0006.mp4')


    ret, frame = cap.read()
    for i in range(68):
        x, y = pts_df.loc[0, [x_cols[i], y_cols[i]]]
        image = cv2.circle(frame, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)

    cv2.imshow('sal', frame)
    cv2.waitKey(0)