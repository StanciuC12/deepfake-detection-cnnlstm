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

    ds_adr = r'E:\FF++'
    csv_path = r'C:\Users\user\Desktop\ML\AI4Media\Datasets\processed_ff'
    save_path = r'E:\saved_img_ff'

    # Points for contour
    x_cols_border = [' x_' + str(x) for x in range(17)] + [' x_' + str(x) for x in range(26, 16, -1)]
    y_cols_border = [' y_' + str(x) for x in range(17)] + [' y_' + str(x) for x in range(26, 16, -1)]

    # Points for eyes
    x_cols_eyes = [' x_' + str(x) for x in [0, 29, 16, 25, 18]]
    y_cols_eyes = [' y_' + str(x) for x in [0, 29, 16, 25, 18]]

    # Points for mouth
    x_cols_mouth = [' x_' + str(x) for x in range(48, 60)]
    y_cols_mouth = [' y_' + str(x) for x in range(48, 60)]

    # Points for nose
    x_cols_nose = [' x_' + str(x) for x in [31, 50, 52, 35, 42, 22, 21, 39]]
    y_cols_nose = [' y_' + str(x) for x in [31, 50, 52, 35, 42, 22, 21, 39]]

    main_dir = ds_adr
    dirs = [x for x in os.listdir(main_dir) if '.' not in x]
    length = 0
    for dir in dirs:
        files_in_dir = os.listdir(os.path.join(main_dir, dir))
        length += len(files_in_dir)

    print('TODO: ', length)
    total_time = 0
    done = 0
    problems_videos = 0
    problem_video_names = []
    for folder in [x for x in os.listdir(ds_adr) if '.' not in x]:
        ds_folder_path = os.path.join(ds_adr, folder)
        for video in os.listdir(ds_folder_path):

            t1 = time.time()

            save_path_folder = os.path.join(save_path, video.split('.')[0])

            try:
                os.mkdir(save_path_folder)
            except:
                continue
            try:
                pts_df = pd.read_csv(os.path.join(csv_path, video.split('.')[0] + '.csv'))
                poz_cols = [x for x in pts_df.columns if x.startswith(' x') or x.startswith(' y')]
                pts_df.loc[:, poz_cols] = pts_df.loc[:, poz_cols].astype(int)
            except:
                continue

            cap = cv2.VideoCapture(os.path.join(os.path.join(ds_adr, folder), video))
            ret = True
            i = 0
            problems = 0
            while ret:

                if problems > 10:
                    problems = 0
                    problems_videos += 1
                    problem_video_names.append(video)
                    break

                try:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if i % 5 == 0 and ret:

                        face_borders = cv2.convexHull(
                            np.stack([pts_df.loc[i, x_cols_border], pts_df.loc[i, y_cols_border]], axis=1).astype(int))[:, 0, :]

                        # Full face
                        Y, X = skimage.draw.polygon(face_borders[:, 1], face_borders[:, 0])
                        blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        blacked_img[Y, X] = frame[Y, X]
                        cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_fullface.png'), cropped_img)

                        # Eyes
                        eyes_borders = cv2.convexHull(
                            np.stack([pts_df.loc[i, x_cols_eyes], pts_df.loc[i, y_cols_eyes]], axis=1).astype(int))[:, 0, :]
                        Y, X = skimage.draw.polygon(eyes_borders[:, 1], eyes_borders[:, 0])
                        blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        blacked_img[Y, X] = frame[Y, X]
                        cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_eyes.png'), cropped_img)

                        # Mouth
                        mouth_borders = cv2.convexHull(
                            np.stack([pts_df.loc[i, x_cols_mouth], pts_df.loc[i, y_cols_mouth]], axis=1).astype(int))[:, 0, :]
                        Y, X = skimage.draw.polygon(mouth_borders[:, 1], mouth_borders[:, 0])
                        blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        blacked_img[Y, X] = frame[Y, X]
                        cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_mouth.png'), cropped_img)

                        #Rest
                        nose_borders = cv2.convexHull(
                            np.stack([pts_df.loc[i, x_cols_nose], pts_df.loc[i, y_cols_nose]], axis=1).astype(int))[:, 0, :]
                        Y, X = skimage.draw.polygon(nose_borders[:, 1], nose_borders[:, 0])
                        blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        blacked_img[Y, X] = frame[Y, X]
                        cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_nose.png'), cropped_img)
                except:
                    problems += 1
                    if problems == 10:
                        print('PROBLEMS: ', str(problems_videos + 1) + '\n')
                    continue

                i += 1

            done += 1

            t2 = time.time()
            total_time += t2 - t1
            remaining_approx_time_h = int((total_time / done) * (length - done) // 3600)
            remaining_approx_time_min = int(((total_time / done) * (length - done) - (total_time / done) * (length - done) // 3600 * 3600) / 60)
            print('=======================================================================\n' + str(done / length * 100)
                  + '% DONE, remaining time: ' + str(remaining_approx_time_h) + 'h' + str(remaining_approx_time_min) + 'm')


print('\n\n\n\n\nPROBLEMS:')
print(problem_video_names)


# f= r'E:\saved_img_dfdc'
# for folder in ['cvdrzlkwdf.mp4', 'cxgieqcozm.mp4', 'demuhxssgl.mp4', 'dgyeaodnzw.mp4', 'idmruoiylw.mp4', 'iurmztyqed.mp4', 'kbssbcdcxx.mp4', 'uakmltagvm.mp4', 'vopokawkip.mp4', 'vqvmlwhabu.mp4', 'wdcrhqtjhf.mp4', 'xawrxpyoau.mp4', 'krxoipopjy.mp4', 'znafprdfbj.mp4', 'fgqwibrpov.mp4', 'jgfirmaztx.mp4', 'jytpyhrwho.mp4', 'lisahsoohl.mp4', 'npgkgevhhw.mp4', 'swvnlemzis.mp4', 'xdwvlfrywz.mp4', 'lyssnzrksj.mp4', 'pggklhbzzr.mp4', 'bwgfsespvf.mp4', 'gcfpxweapn.mp4', 'hvdamhdnvs.mp4', 'iadswupprp.mp4', 'iqaylplffo.mp4', 'jkzbfceoyo.mp4', 'jukglapcgp.mp4', 'kmtfxfdyoj.mp4', 'lihrgmqrff.mp4', 'qbruydwzcc.mp4', 'qehxxibphk.mp4', 'ulubxioaxe.mp4', 'ycvqznokqy.mp4', 'yqnsdfwelt.mp4', 'yxbfjeoniv.mp4', 'iiuqymzjbp.mp4', 'qpmovegayu.mp4', 'xbvjrriwxn.mp4', 'fgcxqtibav.mp4', 'jdyrbxmpsc.mp4', 'szuoacauxp.mp4', 'crkcsrelev.mp4', 'phovymgglb.mp4', 'suhuilfonk.mp4', 'bmrwcqrpyp.mp4', 'boxalsxdzk.mp4', 'bxkvcjrywe.mp4', 'eqaqgiolzr.mp4', 'fjdjdtzfdt.mp4', 'frltisynpx.mp4', 'gwoqohzfgp.mp4', 'gxriengaro.mp4', 'hjlqfygamf.mp4', 'isossxthze.mp4', 'iwsmufejze.mp4', 'jcdwcvhiic.mp4', 'lkgrxiicog.mp4', 'qivdijhwng.mp4', 'qttjeebsxj.mp4', 'rfpjvmqzre.mp4', 'rqcifcowbd.mp4', 'sjtuoasekc.mp4', 'spxsiabtfa.mp4', 'vlkpgibhgk.mp4', 'vyfrgvajts.mp4', 'wdewictsdp.mp4', 'wrvuuxdxbe.mp4', 'xfaoigampc.mp4', 'zraldwcfmh.mp4', 'jykrbbnxhw.mp4', 'zffnhhhwes.mp4', 'kyueoygidm.mp4', 'bruayouvht.mp4', 'cnrodkfuay.mp4', 'gszytodwnu.mp4', 'juavnhcgyl.mp4', 'juxtasggsh.mp4', 'matbflakmq.mp4', 'mbsrhidckc.mp4', 'oowwxjzmaz.mp4', 'oynljxwyps.mp4', 'pzrhmglghk.mp4', 'qposgsibtu.mp4', 'qzkzkxdvva.mp4', 'rpzmrnykdv.mp4', 'tqcvzxsmuc.mp4', 'upntjwlcjh.mp4', 'vlkbweamqm.mp4', 'xiqbkvdpzy.mp4', 'akrubjfzzc.mp4', 'byiqfuoxfa.mp4', 'ekzlzfmhoo.mp4', 'fktxniwzxe.mp4', 'fszexmwczt.mp4', 'gsxmjjhtft.mp4', 'heoufzaddn.mp4', 'kifzxbsnku.mp4', 'krnnelfkhu.mp4', 'spxyvjkiso.mp4', 'ufoipgblmn.mp4', 'wedzanatii.mp4', 'xtpeyyltfi.mp4', 'arjvscherh.mp4', 'arvqnhhwpm.mp4', 'chkyctpgjr.mp4', 'ggjcrroblk.mp4', 'iwladlmomt.mp4', 'jswunuyhcq.mp4', 'otumyuuzkl.mp4', 'pslnizmjib.mp4', 'pxncrwhyia.mp4', 'rgbdumupgz.mp4', 'wfmxpuzxwe.mp4', 'xikpwmuvxy.mp4', 'zarlwbxxug.mp4', 'glivuoahbz.mp4', 'lwywesagym.mp4', 'hiclqqebky.mp4', 'jrphozmnov.mp4', 'svkgvesvdy.mp4', 'cfwxrlwkmn.mp4', 'kjawzgnwrn.mp4', 'qcppqcosbq.mp4', 'thaionimas.mp4', 'vhrzdxqvwp.mp4']:
#
#     try:
#         os.rmdir(os.path.join(f, folder.split('.')[0]))
#     except:
#         print('sal')