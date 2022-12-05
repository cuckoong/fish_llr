# fish_llr

# Project Structure:
   - fish_llr
     - Data (contains all raw data files, with burst level: 4)
       - Quantization (contains all Quantization data files, including Quantization data and labels)
       - Tracking (contains all Tracking data files, including Quantization data and labels)

     - src (contains all source code files)
       - ml_generate_label.py (generate labels for Quantization and Tracking data), control: 0, exp: 1, null: -1
       - ml_Quan_Data_preprocessing.py (extract bout rest/active features with different window size 
       (1min, 2min, 30min post light stimulus), and save data in Processed_data/quantization/batch{}/features folder)
       - ml_classification.py (clustering data with features calculated from Quan_Data_preprocessing.py), results saved in \
       - Analysis_Results/ML_results/{fish_type}/Quan_Data_Classification/acc.csv
       - Visualize.py (visualize data with features calculated from classification results.

     - stats_analysis
       - stats_data_cleaning.py (cleaning burst data, and save data in Processed_data/quantization/Tg/stat_data folder)
       - stats_visualize_burst.R (visualize burst level data, save figures in Figures/Stats/Quantization/Tg)
       
     - Processed_data (contains all processed data files)
       - quantization (contains all Quantization data files, including Quantization data and labels)
       - tracking (contains all Tracking data files, including Quantization data and labels)



ML
1. generate label with ml_generate_label.py and save in Data/Quantization/Tg/{}W-batch{}/labels
2. extract bout rest/active features with ml_Quan_Data_preprocessing.py
3. do classification with ml_classification.py
4. Visualize classification results with ml_acc_visualize_transgenic.py

Statistics for burst duration
1. generate label with ml_generate_label.py and save in Data/Quantization/Tg/{}W-batch{}/{}.npy
2. get burst duration from data and check if fish do not move all the time,  by stats_data_cleaning_burst.py
3. visualize burst duration with stats_visualize_burst.R, result in Figures/Stats/Quantization/Tg/all/scale/batch{}, 
  all means all activity (burst + mid), burst mean only burst, raw is raw, scale is min_max scaling.
4. visualize burst duration with stats_visualize_burst.R, result in Figures/Stats/Quantization/Tg/all/scale/batches, 
  all means all activity (burst + mid), burst mean only burst, raw is raw, scale is min_max scaling.

Statistics for swimming distance
1. generate label with ml_generate_label.py and save in Data/Quantization/Tg/{}W-batch{}/{}.npy
2. get swimming distance (sum) from data and summarise data into one file, by stats_data_cleaning_distance.py, 
   results in Processed_data/tracking/Tg/stat_data named as all_1.2w_60h_batch1_burst4.csv
3. Visualize swimming distance with stats_visualize_distance.R, result in Figures/Stats/Tracking/Tg/all/scale/batch{}, 
   all means all activity (mid + large),  raw is raw, scale is min_max scaling.
4. Visualize batches with stats_visualize_distance_batch.R, result in Figures/Stats/Tracking/Tg/all/scale/batches, 
   all means all activity (mid + large),  raw is raw, scale is min_max scaling.
