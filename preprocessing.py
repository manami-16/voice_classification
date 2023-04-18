import os
import sys
import argparse
from pydub import AudioSegment
import glob
import shutil
import pandas as pd
from scipy.io.wavfile import read
import numpy as np

def convert_m4a_to_wav():
    m4a_dir = './m4a_data/'
    extension_list = ('*.m4a', '*.flv')
    os.chdir(m4a_dir)
    export_dir = '../wav_data/'
    
    for extension in extension_list:
        for audio in glob.glob(extension):
            wav_filename = os.path.splitext(os.path.basename(audio))[0] + '.wav'
            wav_filename = wav_filename.split('_')[0] + '_' + wav_filename.split('_')[-1]
            AudioSegment.from_file(audio).export(export_dir + wav_filename, format='wav')
    
#     os.close(m4a_dir)
    return None

def make_dataframe():
    path = '../wav_data/'
    dir_list = os.listdir(path)
    data = {'filename': [], 
            'class': [], 
            'classID': []}

    for file in dir_list:
        _file = file.split('_')
        if _file[0] == 'angry':
            data['filename'].append(_file[0] + _file[-1])
            data['class'].append('angry')
            data['classID'].append(0)
            
        elif _file[0] == 'happy':
            data['filename'].append(_file[0] + _file[-1])
            data['class'].append('happy')
            data['classID'].append(1)
    
#     os.close()
    print(data)
    df = pd.DataFrame(data)
    df.to_csv('../dataframe.csv')
    
    return df

def remove_silence(alpha=20):
    
    ## retreive all data under wav_data
    ## extract where the absolute intensity is greater than alpha = 20
    path = '../wav_data/'
    dir_list = os.listdir(path)
    
    trimmed_data = []
    
    for i in range(len(dir_list)):
        if dir_list[i] != '.ipynb_checkpoints':
            data = read(path + dir_list[i])[1]
            data = np.array(data)
            num_slience = np.count_nonzero(abs(data) < 20)
            ratio = 100 * num_slience / len(data)
            print(f'{i} -- {ratio:.2f}% of data will be removed')
            
            data = data[abs(data) > alpha]
            filename = '../trimmed_data/' + dir_list[i].split('.')[0] + '.npy'
            np.save(file=filename, arr=data)
#     os.close(path)
    return None

def normalize_data():
    
    ## min/max normalization
    path = '../trimmed_data/'
    dir_list = os.listdir(path)
    
    for i in range(len(dir_list)):
        if dir_list[i] != '.ipynb_checkpoints':
            data = np.load(path + dir_list[i])
            
            diff = abs(max(data) - min(data))
            min_data = min(data)
            
            normalized_data = []
            for d in data:
                _d = float((d - min_data) / diff)
                normalized_data.append(_d)

            normalized_data = np.array(normalized_data)
            filename = path + dir_list[i].split('.')[0] + '.npy'
            np.save(file=filename, arr=normalized_data)
#     os.close()
    return None

    
    