import pandas as pd
import datetime

def convert_to_time(milliseconds):
    seconds = milliseconds // 1000000
    milliseconds %= 1000000
    return datetime.datetime.utcfromtimestamp(seconds).strftime('%M:%S.') + f'{milliseconds:06d}'

df = pd.read_csv('/home/radar/Documents/camera/RADAR_20230527_171541@20230527_171541794532_rx_radar_data_1_Radar_Status.csv')

df['new_timestamps'] = df['Timestamps'].apply(convert_to_time)
df
df.to_csv('new_time.csv',index = False)