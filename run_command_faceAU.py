import os

main_dir = r'E:\FF++'
dirs = [x for x in os.listdir(main_dir) if '.' not in x]

#os.system(r'cd C:\Users\user\Desktop\ML\AI4Media\Code\OpenFace20\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64')
os.chdir(r'C:\Users\user\Desktop\ML\AI4Media\Code\OpenFace20\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64')

#dir_folders = [x for x in os.listdir("E:\processed_DFDC") if '.' not in x]

already_processed = [x.split('.')[0] for x in os.listdir(r'C:\Users\user\Desktop\ML\AI4Media\Datasets\processed_ff') if x.endswith('.csv')]
i = 0
length = 0
for dir in dirs:

    files_in_dir = os.listdir(os.path.join(main_dir, dir))
    length += len(files_in_dir)

done = 0
for dir in dirs:

    files_in_dir = os.listdir(os.path.join(main_dir, dir))

    i = 0
    file_adresses = []
    last = False
    for file in files_in_dir:

        name_wo_end = file.split('.')[0]
        if file == files_in_dir[-1]:
            last = True

        if name_wo_end not in already_processed:

            i += 1
            if last:
                i = 100
            command = 'FeatureExtraction.exe'
            file_adr = os.path.join(os.path.join(main_dir, dir), file)
            file_adresses.append(file_adr)

            if i % 100 == 0:

                file_adr = ' -f '.join(file_adresses)
                command = command + ' -f ' + file_adr + r' -out_dir "C:\Users\user\Desktop\ML\AI4Media\Datasets\processed_ff" -2Dfp '
                print('sal')

                try:
                    os.system(command)
                    done += 100
                except:
                    print('Nu a mers comanda')
                    continue

                file_adresses = []
                i = 0

        else:
            done += 1

        print('=======================================================================\n' + str(done/length * 100)
              + '% DONE\n' + '=======================================================================')



