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

    ds_adr = r'F:\ff++\original_sequences'
    csv_path = r'C:\Users\user\Desktop\ML\AI4Media\Datasets\processed_ff' #r'E:\processed_celebdf' #r'C:\Users\user\Desktop\ML\AI4Media\Datasets\processed_ff'
    save_path = r'F:\ff++\saved_images' #r'E:\saved_img_ff'

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
    for folder in [x for x in os.listdir(ds_adr) if '.' not in x][::-1]:
        ds_folder_path = os.path.join(ds_adr, folder)

        try:
            os.mkdir(os.path.join(save_path, folder))
        except:
            print('Dir already existat')

        for video in os.listdir(ds_folder_path):

            t1 = time.time()

            save_path_folder = os.path.join(save_path, folder, video.split('.')[0])

            try:
                os.mkdir(save_path_folder)
            except:
                print('Dir already exists')
                continue

            try:
                pts_df = pd.read_csv(os.path.join(csv_path, video.split('.')[0] + '.csv'))
                poz_cols = [x for x in pts_df.columns if x.startswith(' x') or x.startswith(' y')]
                pts_df.loc[:, poz_cols] = pts_df.loc[:, poz_cols].astype(int)
            except:
                print('Can not ')
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
                    if ret:

                        face_borders = cv2.convexHull(
                            np.stack([pts_df.loc[i, x_cols_border], pts_df.loc[i, y_cols_border]], axis=1).astype(int))[:, 0, :]

                        # Full face
                        Y, X = skimage.draw.polygon(face_borders[:, 1], face_borders[:, 0])
                        blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        blacked_img[Y, X] = frame[Y, X]
                        cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_fullface.png'), cropped_img)

                        # # Eyes
                        # eyes_borders = cv2.convexHull(
                        #     np.stack([pts_df.loc[i, x_cols_eyes], pts_df.loc[i, y_cols_eyes]], axis=1).astype(int))[:, 0, :]
                        # Y, X = skimage.draw.polygon(eyes_borders[:, 1], eyes_borders[:, 0])
                        # blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        # blacked_img[Y, X] = frame[Y, X]
                        # cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        # cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_eyes.png'), cropped_img)
                        #
                        # # Mouth
                        # mouth_borders = cv2.convexHull(
                        #     np.stack([pts_df.loc[i, x_cols_mouth], pts_df.loc[i, y_cols_mouth]], axis=1).astype(int))[:, 0, :]
                        # Y, X = skimage.draw.polygon(mouth_borders[:, 1], mouth_borders[:, 0])
                        # blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        # blacked_img[Y, X] = frame[Y, X]
                        # cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        # cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_mouth.png'), cropped_img)
                        #
                        # #Rest
                        # nose_borders = cv2.convexHull(
                        #     np.stack([pts_df.loc[i, x_cols_nose], pts_df.loc[i, y_cols_nose]], axis=1).astype(int))[:, 0, :]
                        # Y, X = skimage.draw.polygon(nose_borders[:, 1], nose_borders[:, 0])
                        # blacked_img = np.zeros(frame.shape, dtype=np.uint8)
                        # blacked_img[Y, X] = frame[Y, X]
                        # cropped_img = crop_square(blacked_img, face_borders, final_dim=(299, 299))
                        # cv2.imwrite(os.path.join(save_path_folder, 'frame_' + str(i) + '_nose.png'), cropped_img)
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
# for folder in ['638_640.mp4', '725_724.mp4', '638_640.mp4', '725_724.mp4', '001_870.mp4', '003_000.mp4', '004_982.mp4', '005_010.mp4', '006_002.mp4', '013_883.mp4', '014_790.mp4', '015_919.mp4', '018_019.mp4', '020_344.mp4', '022_489.mp4', '023_923.mp4', '024_073.mp4', '026_012.mp4', '027_009.mp4', '029_048.mp4', '030_193.mp4', '032_944.mp4', '034_590.mp4', '036_035.mp4', '038_125.mp4', '039_058.mp4', '041_063.mp4', '042_084.mp4', '043_110.mp4', '044_945.mp4', '047_862.mp4', '049_946.mp4', '050_059.mp4', '053_095.mp4', '054_071.mp4', '055_147.mp4', '061_080.mp4', '062_066.mp4', '065_089.mp4', '067_025.mp4', '068_028.mp4', '070_057.mp4', '072_037.mp4', '074_825.mp4', '075_977.mp4', '076_079.mp4', '082_103.mp4', '087_081.mp4', '088_060.mp4', '090_086.mp4', '093_121.mp4', '096_101.mp4', '097_033.mp4', '098_092.mp4', '100_077.mp4', '102_114.mp4', '105_180.mp4', '106_198.mp4', '108_052.mp4', '109_107.mp4', '111_094.mp4', '115_939.mp4', '116_091.mp4', '117_217.mp4', '118_120.mp4', '119_123.mp4', '122_144.mp4', '124_085.mp4', '126_104.mp4', '129_127.mp4', '132_007.mp4', '135_880.mp4', '138_142.mp4', '139_130.mp4', '143_140.mp4', '148_133.mp4', '150_153.mp4', '152_149.mp4', '159_175.mp4', '161_141.mp4', '163_031.mp4', '165_137.mp4', '167_166.mp4', '171_173.mp4', '174_964.mp4', '177_211.mp4', '178_598.mp4', '183_253.mp4', '186_170.mp4', '188_191.mp4', '189_200.mp4', '190_176.mp4', '192_134.mp4', '195_442.mp4', '199_181.mp4', '202_348.mp4', '203_201.mp4', '205_184.mp4', '207_908.mp4', '208_215.mp4', '209_016.mp4', '212_179.mp4', '213_083.mp4', '216_164.mp4', '218_239.mp4', '222_168.mp4', '223_586.mp4', '224_197.mp4', '225_151.mp4', '226_491.mp4', '227_169.mp4', '228_289.mp4', '230_204.mp4', '234_187.mp4', '235_194.mp4', '236_237.mp4', '241_210.mp4', '242_182.mp4', '243_156.mp4', '245_157.mp4', '248_232.mp4', '250_461.mp4', '254_261.mp4', '256_146.mp4', '257_420.mp4', '260_331.mp4', '262_301.mp4', '263_284.mp4', '264_271.mp4', '266_252.mp4', '267_286.mp4', '269_268.mp4', '270_297.mp4', '272_396.mp4', '273_807.mp4', '274_412.mp4', '276_185.mp4', '278_306.mp4', '280_249.mp4', '282_238.mp4', '285_136.mp4', '288_321.mp4', '290_240.mp4', '291_874.mp4', '292_294.mp4', '295_099.mp4', '296_293.mp4', '298_279.mp4', '299_145.mp4', '300_304.mp4', '303_309.mp4', '308_388.mp4', '310_196.mp4', '311_387.mp4', '313_283.mp4', '314_347.mp4', '318_334.mp4', '323_302.mp4', '328_320.mp4', '329_327.mp4', '330_162.mp4', '332_051.mp4', '335_277.mp4', '336_338.mp4', '337_522.mp4', '339_392.mp4', '340_341.mp4', '342_416.mp4', '343_363.mp4', '345_259.mp4', '346_351.mp4', '350_349.mp4', '352_319.mp4', '353_383.mp4', '356_324.mp4', '357_432.mp4', '359_317.mp4', '360_437.mp4', '361_448.mp4', '366_473.mp4', '367_371.mp4', '369_316.mp4', '376_381.mp4', '377_333.mp4', '379_158.mp4', '380_358.mp4', '382_409.mp4', '384_932.mp4', '385_414.mp4', '386_154.mp4', '389_480.mp4', '394_373.mp4', '395_401.mp4', '397_602.mp4', '402_453.mp4', '405_393.mp4', '406_391.mp4', '407_374.mp4', '410_411.mp4', '413_372.mp4', '417_496.mp4', '418_507.mp4', '419_824.mp4', '421_423.mp4', '425_485.mp4', '426_287.mp4', '427_637.mp4', '429_404.mp4', '430_459.mp4', '434_438.mp4', '435_456.mp4', '436_526.mp4', '440_364.mp4', '441_439.mp4', '444_655.mp4', '446_667.mp4', '447_431.mp4', '449_451.mp4', '455_471.mp4', '457_398.mp4', '458_722.mp4', '463_464.mp4', '465_482.mp4', '466_428.mp4', '467_462.mp4', '468_470.mp4', '472_511.mp4', '474_281.mp4', '475_265.mp4', '476_400.mp4', '483_370.mp4', '484_415.mp4', '487_477.mp4', '488_399.mp4', '492_325.mp4', '493_538.mp4', '494_445.mp4', '495_512.mp4', '497_403.mp4', '498_433.mp4', '500_592.mp4', '501_326.mp4', '506_478.mp4', '510_528.mp4', '513_305.mp4', '514_443.mp4', '516_555.mp4', '518_131.mp4', '519_515.mp4', '521_517.mp4', '523_541.mp4', '525_509.mp4', '527_454.mp4', '529_633.mp4', '531_549.mp4', '533_450.mp4', '534_490.mp4', '536_540.mp4', '539_499.mp4', '542_520.mp4', '544_532.mp4', '547_574.mp4', '550_452.mp4', '552_851.mp4', '553_545.mp4', '556_588.mp4', '559_543.mp4', '560_557.mp4', '562_626.mp4', '565_589.mp4', '567_606.mp4', '568_628.mp4', '572_554.mp4', '576_155.mp4', '577_593.mp4', '580_524.mp4', '581_697.mp4', '582_172.mp4', '583_558.mp4', '584_823.mp4', '585_599.mp4', '587_535.mp4', '591_605.mp4', '594_530.mp4', '597_595.mp4', '600_505.mp4', '601_653.mp4', '603_575.mp4', '609_596.mp4', '612_702.mp4', '613_685.mp4', '614_616.mp4', '615_687.mp4', '617_566.mp4', '618_629.mp4', '619_620.mp4', '621_546.mp4', '622_647.mp4', '623_630.mp4', '624_570.mp4', '625_650.mp4', '627_658.mp4', '631_551.mp4', '632_548.mp4', '636_578.mp4', '638_640.mp4', '640_638.mp4', '641_662.mp4', '642_635.mp4', '643_646.mp4', '644_657.mp4', '645_688.mp4', '651_835.mp4', '652_773.mp4', '654_648.mp4', '663_231.mp4', '664_668.mp4', '665_679.mp4', '666_656.mp4', '669_682.mp4', '670_661.mp4', '672_720.mp4', '675_608.mp4', '677_671.mp4', '678_460.mp4', '680_486.mp4', '683_607.mp4', '686_696.mp4', '690_689.mp4', '692_610.mp4', '694_767.mp4', '698_693.mp4', '699_734.mp4', '700_813.mp4', '701_579.mp4', '703_604.mp4', '704_723.mp4', '705_707.mp4', '706_479.mp4', '709_390.mp4', '710_788.mp4', '711_681.mp4', '712_716.mp4', '717_684.mp4', '721_715.mp4', '724_725.mp4', '725_724.mp4', '726_713.mp4', '727_729.mp4', '728_673.mp4', '732_691.mp4', '733_935.mp4', '737_719.mp4', '739_865.mp4', '740_796.mp4', '741_731.mp4', '743_750.mp4', '744_674.mp4', '746_571.mp4', '748_355.mp4', '749_659.mp4', '751_752.mp4', '755_759.mp4', '756_503.mp4', '757_573.mp4', '760_611.mp4', '761_766.mp4', '762_832.mp4', '764_850.mp4', '765_867.mp4', '768_793.mp4', '769_784.mp4', '772_708.mp4', '774_735.mp4', '775_742.mp4', '776_676.mp4', '777_745.mp4', '778_798.mp4', '779_794.mp4', '783_916.mp4', '786_819.mp4', '787_782.mp4', '789_753.mp4', '791_770.mp4', '792_903.mp4', '797_844.mp4', '800_840.mp4', '801_855.mp4', '803_017.mp4', '804_738.mp4', '805_011.mp4', '806_781.mp4', '808_829.mp4', '809_799.mp4', '815_730.mp4', '816_649.mp4', '818_820.mp4', '822_244.mp4', '826_833.mp4', '827_817.mp4', '828_830.mp4', '831_508.mp4', '834_852.mp4', '836_950.mp4', '838_810.mp4', '841_639.mp4', '842_714.mp4', '843_859.mp4', '847_906.mp4', '849_771.mp4', '854_747.mp4', '858_861.mp4', '863_853.mp4', '864_839.mp4', '866_878.mp4', '868_949.mp4', '869_780.mp4', '871_814.mp4', '872_873.mp4', '881_856.mp4', '882_952.mp4', '885_802.mp4', '886_877.mp4', '887_275.mp4', '888_937.mp4', '889_045.mp4', '890_837.mp4', '891_876.mp4', '892_112.mp4', '893_913.mp4', '894_848.mp4', '895_915.mp4', '896_128.mp4', '897_969.mp4', '898_922.mp4', '902_901.mp4', '904_046.mp4', '905_860.mp4', '907_795.mp4', '910_911.mp4', '912_927.mp4', '914_899.mp4', '917_924.mp4', '920_811.mp4', '921_569.mp4', '925_933.mp4', '928_160.mp4', '930_763.mp4', '934_918.mp4', '936_931.mp4', '938_987.mp4', '940_941.mp4', '943_942.mp4', '947_951.mp4', '957_959.mp4', '958_956.mp4', '961_069.mp4', '963_879.mp4', '965_948.mp4', '966_988.mp4', '968_884.mp4', '970_973.mp4', '971_564.mp4', '972_718.mp4', '974_953.mp4', '975_978.mp4', '976_954.mp4', '979_875.mp4', '980_992.mp4', '983_113.mp4', '984_967.mp4', '985_981.mp4', '990_008.mp4', '993_989.mp4', '994_986.mp4', '995_233.mp4', '996_056.mp4', '997_040.mp4', '998_561.mp4', '638_640.mp4', '725_724.mp4']:
#
#     try:
#         os.rmdir(os.path.join(f, folder.split('.')[0]))
#     except:
#         print('sal')


# # PROBLEMS:
# a = ['638_640.mp4', '725_724.mp4', '638_640.mp4', '725_724.mp4', '001_870.mp4', '003_000.mp4', '004_982.mp4', '005_010.mp4', '006_002.mp4', '013_883.mp4', '014_790.mp4', '015_919.mp4', '018_019.mp4', '020_344.mp4', '022_489.mp4', '023_923.mp4', '024_073.mp4', '026_012.mp4', '027_009.mp4', '029_048.mp4', '030_193.mp4', '032_944.mp4', '034_590.mp4', '036_035.mp4', '038_125.mp4', '039_058.mp4', '041_063.mp4', '042_084.mp4', '043_110.mp4', '044_945.mp4', '047_862.mp4', '049_946.mp4', '050_059.mp4', '053_095.mp4', '054_071.mp4', '055_147.mp4', '061_080.mp4', '062_066.mp4', '065_089.mp4', '067_025.mp4', '068_028.mp4', '070_057.mp4', '072_037.mp4', '074_825.mp4', '075_977.mp4', '076_079.mp4', '082_103.mp4', '087_081.mp4', '088_060.mp4', '090_086.mp4', '093_121.mp4', '096_101.mp4', '097_033.mp4', '098_092.mp4', '100_077.mp4', '102_114.mp4', '105_180.mp4', '106_198.mp4', '108_052.mp4', '109_107.mp4', '111_094.mp4', '115_939.mp4', '116_091.mp4', '117_217.mp4', '118_120.mp4', '119_123.mp4', '122_144.mp4', '124_085.mp4', '126_104.mp4', '129_127.mp4', '132_007.mp4', '135_880.mp4', '138_142.mp4', '139_130.mp4', '143_140.mp4', '148_133.mp4', '150_153.mp4', '152_149.mp4', '159_175.mp4', '161_141.mp4', '163_031.mp4', '165_137.mp4', '167_166.mp4', '171_173.mp4', '174_964.mp4', '177_211.mp4', '178_598.mp4', '183_253.mp4', '186_170.mp4', '188_191.mp4', '189_200.mp4', '190_176.mp4', '192_134.mp4', '195_442.mp4', '199_181.mp4', '202_348.mp4', '203_201.mp4', '205_184.mp4', '207_908.mp4', '208_215.mp4', '209_016.mp4', '212_179.mp4', '213_083.mp4', '216_164.mp4', '218_239.mp4', '222_168.mp4', '223_586.mp4', '224_197.mp4', '225_151.mp4', '226_491.mp4', '227_169.mp4', '228_289.mp4', '230_204.mp4', '234_187.mp4', '235_194.mp4', '236_237.mp4', '241_210.mp4', '242_182.mp4', '243_156.mp4', '245_157.mp4', '248_232.mp4', '250_461.mp4', '254_261.mp4', '256_146.mp4', '257_420.mp4', '260_331.mp4', '262_301.mp4', '263_284.mp4', '264_271.mp4', '266_252.mp4', '267_286.mp4', '269_268.mp4', '270_297.mp4', '272_396.mp4', '273_807.mp4', '274_412.mp4', '276_185.mp4', '278_306.mp4', '280_249.mp4', '282_238.mp4', '285_136.mp4', '288_321.mp4', '290_240.mp4', '291_874.mp4', '292_294.mp4', '295_099.mp4', '296_293.mp4', '298_279.mp4', '299_145.mp4', '300_304.mp4', '303_309.mp4', '308_388.mp4', '310_196.mp4', '311_387.mp4', '313_283.mp4', '314_347.mp4', '318_334.mp4', '323_302.mp4', '328_320.mp4', '329_327.mp4', '330_162.mp4', '332_051.mp4', '335_277.mp4', '336_338.mp4', '337_522.mp4', '339_392.mp4', '340_341.mp4', '342_416.mp4', '343_363.mp4', '345_259.mp4', '346_351.mp4', '350_349.mp4', '352_319.mp4', '353_383.mp4', '356_324.mp4', '357_432.mp4', '359_317.mp4', '360_437.mp4', '361_448.mp4', '366_473.mp4', '367_371.mp4', '369_316.mp4', '376_381.mp4', '377_333.mp4', '379_158.mp4', '380_358.mp4', '382_409.mp4', '384_932.mp4', '385_414.mp4', '386_154.mp4', '389_480.mp4', '394_373.mp4', '395_401.mp4', '397_602.mp4', '402_453.mp4', '405_393.mp4', '406_391.mp4', '407_374.mp4', '410_411.mp4', '413_372.mp4', '417_496.mp4', '418_507.mp4', '419_824.mp4', '421_423.mp4', '425_485.mp4', '426_287.mp4', '427_637.mp4', '429_404.mp4', '430_459.mp4', '434_438.mp4', '435_456.mp4', '436_526.mp4', '440_364.mp4', '441_439.mp4', '444_655.mp4', '446_667.mp4', '447_431.mp4', '449_451.mp4', '455_471.mp4', '457_398.mp4', '458_722.mp4', '463_464.mp4', '465_482.mp4', '466_428.mp4', '467_462.mp4', '468_470.mp4', '472_511.mp4', '474_281.mp4', '475_265.mp4', '476_400.mp4', '483_370.mp4', '484_415.mp4', '487_477.mp4', '488_399.mp4', '492_325.mp4', '493_538.mp4', '494_445.mp4', '495_512.mp4', '497_403.mp4', '498_433.mp4', '500_592.mp4', '501_326.mp4', '506_478.mp4', '510_528.mp4', '513_305.mp4', '514_443.mp4', '516_555.mp4', '518_131.mp4', '519_515.mp4', '521_517.mp4', '523_541.mp4', '525_509.mp4', '527_454.mp4', '529_633.mp4', '531_549.mp4', '533_450.mp4', '534_490.mp4', '536_540.mp4', '539_499.mp4', '542_520.mp4', '544_532.mp4', '547_574.mp4', '550_452.mp4', '552_851.mp4', '553_545.mp4', '556_588.mp4', '559_543.mp4', '560_557.mp4', '562_626.mp4', '565_589.mp4', '567_606.mp4', '568_628.mp4', '572_554.mp4', '576_155.mp4', '577_593.mp4', '580_524.mp4', '581_697.mp4', '582_172.mp4', '583_558.mp4', '584_823.mp4', '585_599.mp4', '587_535.mp4', '591_605.mp4', '594_530.mp4', '597_595.mp4', '600_505.mp4', '601_653.mp4', '603_575.mp4', '609_596.mp4', '612_702.mp4', '613_685.mp4', '614_616.mp4', '615_687.mp4', '617_566.mp4', '618_629.mp4', '619_620.mp4', '621_546.mp4', '622_647.mp4', '623_630.mp4', '624_570.mp4', '625_650.mp4', '627_658.mp4', '631_551.mp4', '632_548.mp4', '636_578.mp4', '638_640.mp4', '640_638.mp4', '641_662.mp4', '642_635.mp4', '643_646.mp4', '644_657.mp4', '645_688.mp4', '651_835.mp4', '652_773.mp4', '654_648.mp4', '663_231.mp4', '664_668.mp4', '665_679.mp4', '666_656.mp4', '669_682.mp4', '670_661.mp4', '672_720.mp4', '675_608.mp4', '677_671.mp4', '678_460.mp4', '680_486.mp4', '683_607.mp4', '686_696.mp4', '690_689.mp4', '692_610.mp4', '694_767.mp4', '698_693.mp4', '699_734.mp4', '700_813.mp4', '701_579.mp4', '703_604.mp4', '704_723.mp4', '705_707.mp4', '706_479.mp4', '709_390.mp4', '710_788.mp4', '711_681.mp4', '712_716.mp4', '717_684.mp4', '721_715.mp4', '724_725.mp4', '725_724.mp4', '726_713.mp4', '727_729.mp4', '728_673.mp4', '732_691.mp4', '733_935.mp4', '737_719.mp4', '739_865.mp4', '740_796.mp4', '741_731.mp4', '743_750.mp4', '744_674.mp4', '746_571.mp4', '748_355.mp4', '749_659.mp4', '751_752.mp4', '755_759.mp4', '756_503.mp4', '757_573.mp4', '760_611.mp4', '761_766.mp4', '762_832.mp4', '764_850.mp4', '765_867.mp4', '768_793.mp4', '769_784.mp4', '772_708.mp4', '774_735.mp4', '775_742.mp4', '776_676.mp4', '777_745.mp4', '778_798.mp4', '779_794.mp4', '783_916.mp4', '786_819.mp4', '787_782.mp4', '789_753.mp4', '791_770.mp4', '792_903.mp4', '797_844.mp4', '800_840.mp4', '801_855.mp4', '803_017.mp4', '804_738.mp4', '805_011.mp4', '806_781.mp4', '808_829.mp4', '809_799.mp4', '815_730.mp4', '816_649.mp4', '818_820.mp4', '822_244.mp4', '826_833.mp4', '827_817.mp4', '828_830.mp4', '831_508.mp4', '834_852.mp4', '836_950.mp4', '838_810.mp4', '841_639.mp4', '842_714.mp4', '843_859.mp4', '847_906.mp4', '849_771.mp4', '854_747.mp4', '858_861.mp4', '863_853.mp4', '864_839.mp4', '866_878.mp4', '868_949.mp4', '869_780.mp4', '871_814.mp4', '872_873.mp4', '881_856.mp4', '882_952.mp4', '885_802.mp4', '886_877.mp4', '887_275.mp4', '888_937.mp4', '889_045.mp4', '890_837.mp4', '891_876.mp4', '892_112.mp4', '893_913.mp4', '894_848.mp4', '895_915.mp4', '896_128.mp4', '897_969.mp4', '898_922.mp4', '902_901.mp4', '904_046.mp4', '905_860.mp4', '907_795.mp4', '910_911.mp4', '912_927.mp4', '914_899.mp4', '917_924.mp4', '920_811.mp4', '921_569.mp4', '925_933.mp4', '928_160.mp4', '930_763.mp4', '934_918.mp4', '936_931.mp4', '938_987.mp4', '940_941.mp4', '943_942.mp4', '947_951.mp4', '957_959.mp4', '958_956.mp4', '961_069.mp4', '963_879.mp4', '965_948.mp4', '966_988.mp4', '968_884.mp4', '970_973.mp4', '971_564.mp4', '972_718.mp4', '974_953.mp4', '975_978.mp4', '976_954.mp4', '979_875.mp4', '980_992.mp4', '983_113.mp4', '984_967.mp4', '985_981.mp4', '990_008.mp4', '993_989.mp4', '994_986.mp4', '995_233.mp4', '996_056.mp4', '997_040.mp4', '998_561.mp4', '638_640.mp4', '725_724.mp4']
# print(len(a), len(np.unique(a)))

# a = r'F:\ff++\saved_images'
# folders = os.listdir(a)
# empty = []
# for f in folders:
#     for f_in in os.listdir(os.path.join(a, f)):
#
#         if len(os.listdir(os.path.join(a, f, f_in))) == 0:
#             empty.append(os.path.join(a, f, f_in))
#
# print(empty)
# print(len(empty))
#
# for folder in empty:
#     try:
#         os.rmdir(folder)
#     except:
#         print('sal')