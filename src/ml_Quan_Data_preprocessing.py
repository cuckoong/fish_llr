import os
from utils import *

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')

if __name__ == '__main__':
    # Load data from csv file
    fish_type = 'Tg'  # ['WT', 'Tg']
    radiation = 1
    batch = 2
    hour = 60
    days = [5, 6, 7, 8]
    plates = [1]

    data_dir = os.path.join('Data/Quantization/{}/{}W-batch{}/'.format(fish_type, radiation, batch))
    metric_list = ['frect', 'fredur', 'midct', 'middur', 'burct', 'burdur', 'aname', 'time']

    result_dir = os.path.join('Processed_data/quantization/{}/batch{}/'.format(fish_type, batch))
    for day in days:
        for plate in plates:
            filename = '{}W-{}h-{}dpf-0{}'.format(radiation, hour, day, plate)
            file = os.path.join(data_dir, filename + '.csv')
            # data
            data_cp = pd.read_csv(file)
            data_cp['time'] = data_cp['end'] // 60
            data_metric = data_cp[metric_list]
            # group each 60 seconds into one row
            data_sum = data_metric.groupby(['aname', 'time']).sum().reset_index()
            data_sum.to_csv(os.path.join(result_dir, 'raw_features', filename + '-data.csv'), index=False)
            # labels
            labels = np.load(os.path.join(data_dir, filename + '-label.npy'))
            data_sum.sort_values(['time', 'aname'], inplace=True)
            data = data_sum.values
            data = data.reshape(-1, 96, len(metric_list))
            data = np.transpose(data, (1, 0, 2))

            # remove well without fish
            data = data[labels >= 0, :, :]
            periods = [30]
            # periods = ['baseline', 1, 2, 30]
            for period in periods:  # time (minutes) after on/off stimulus
                features_df = get_features(data, period=period, mode='quantization')
                features_df['label'] = labels[labels >= 0]
                features_df.to_csv(os.path.join(result_dir, 'features',
                                                filename + '-{}-min.csv'.format(period)), index=False)
