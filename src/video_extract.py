import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')

if __name__ == '__main__':
    POWER = 1.2
    BATCH = 2
    DAY = 8

    video_path = f'Data/Quantization/Tg/{POWER}W-batch{BATCH}/{POWER}W-60h-{DAY}dpf-01.avi'
    background_mask_folder = f'Figures/background_image/{POWER}W-batch{BATCH}/{POWER}W-60h-{DAY}dpf-01'

    # start time and end time
    stim_times = [1800, 3600, 5400, 7200]
    start_times = [(t - 30) * 1000 for t in stim_times]  # pre_duration = 30s
    end_times = [(t + 30) * 1000 for t in stim_times]  # post_duration = 30s

    for i in range(len(start_times)):
        # get the start time and end time
        start_time = start_times[i]
        end_time = end_times[i]

        # create an array to store the intensity values
        intensity = {}

        # load the video
        cap = cv2.VideoCapture(video_path)

        # check sample rate
        fps = cap.get(cv2.CAP_PROP_FPS)

        # read the frame between start time and end time
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time - 1000)

        while cap.isOpened():
            ret, frame = cap.read()

            # check if the frame is empty
            if not ret:
                break

            # convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # add masks to the frame (masks in the background folder)
            masks = os.listdir(background_mask_folder)
            masks = [mask for mask in masks if mask.endswith('.png')]

            # apply the masks
            for mask_png in masks:
                # read image from bin8 png
                mask = cv2.imread(f'{background_mask_folder}/{mask_png}')
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_name = mask_png.split('.')[0].split('_')[-1]

                # invert the mask
                mask = cv2.bitwise_not(mask)

                # any pixel with value lower than 255 will be set to 0
                mask[mask < 255] = 0

                # apply the mask
                gray_masked = cv2.bitwise_and(gray, mask)

                # log the intensity
                masked_intensity = np.sum(gray_masked) / np.sum(mask / 255)

                if mask_name not in intensity:
                    intensity[mask_name] = [masked_intensity]
                else:
                    intensity[mask_name].append(masked_intensity)

            if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > (end_time + 1000):
                # release the video capture object
                cap.release()
                cv2.destroyAllWindows()
                break

            # need to skip frames to get 1 frame per second
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + fps)

        # FIGURE path
        figure_path = f'Figures/light_intensity/Tg/{POWER}W-batch{BATCH}/{POWER}W-60h-{DAY}dpf-01/'
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        # visualize the intensity for each mask
        plt.figure(figsize=(10, 5))
        for mask_name in intensity:
            plt.plot(intensity[mask_name], label=mask_name)
        plt.savefig(os.path.join(figure_path, f'{start_time}_{end_time}_raw.png'))

        plt.figure(figsize=(10, 5))
        for mask_name in intensity:
            plt.plot(intensity[mask_name] - intensity[mask_name][0], label=mask_name)
        plt.savefig(os.path.join(figure_path, f'{start_time}_{end_time}_rm_baseline.png'))

        if not os.path.exists(f'Processed_data/light_intensity/Tg/{POWER}W-batch{BATCH}/{POWER}W-60h-{DAY}dpf-01'):
            os.makedirs(f'Processed_data/light_intensity/Tg/{POWER}W-batch{BATCH}/{POWER}W-60h-{DAY}dpf-01')

        # save the intensity to pickle
        # convert data to dataframe
        df_dict = {
            'animal_id': [],
            'end': [],
            'light_intensity': []
        }
        for key in intensity.keys():
            df_dict['animal_id'].extend([f'{BATCH}-1-{key}'] * len(intensity[key]))
            df_dict['end'].extend(np.arange(stim_times[i] - 30, stim_times[i] + 31, 1))
            df_dict['light_intensity'].extend(intensity[key])

        df = pd.DataFrame(df_dict)

        df.to_csv(f'Processed_data/light_intensity/Tg/{POWER}W-batch{BATCH}/{POWER}W-60h-{DAY}dpf-01/'
                  f'{start_time}_{end_time}.csv', index=False)