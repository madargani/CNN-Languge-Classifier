import pandas as pd
import os
import librosa
import numpy as np

SAMPLE_RATE = 16000

# preprocessing train data
# load data, trim silence, save mel spectrograms
for language in ('ar', 'en', 'zh-TW'):
    paths = pd.read_csv(f'data/{language}/validated.tsv', sep='\t')['path']
    
    for idx, path in paths.items():
        if len(os.listdir(f'mel_specs/{language}')) > 5000:
            break
        
        y, sr = librosa.load(f'data/{language}/clips/{path}', sr=SAMPLE_RATE, mono=True)
        y, index = librosa.effects.trim(y, top_db=25)
        if (len(y) < 16000):
            print('skipping')
            continue
        
        for i in range(len(y) // sr):
            S = librosa.feature.melspectrogram(
                y=y[i * sr: (i + 1) * sr], sr=SAMPLE_RATE,
                n_fft=512, win_length=400, hop_length=160)
            S = librosa.power_to_db(S)
            np.save(f'mel_specs/{language}/{idx}_{i}.npy', S)
            
# print file distribution
ar_files = os.listdir('mel_specs/ar')
en_files = os.listdir('mel_specs/en')
zh_files = os.listdir('mel_specs/zh-TW')

print(f'Arabic: {len(ar_files)}')
print(f'English: {len(en_files)}')
print(f'Chinese: {len(zh_files)}')