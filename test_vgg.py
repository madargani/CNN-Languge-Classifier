import torch
from torchvision.models import vgg16
import numpy as np
import librosa
import cv2
from Data import AudioDataset
from IPython.display import Audio, display

# load vgg model
vgg_model = vgg16()
vgg_model.load_state_dict(torch.load('models/vgg_1.pth'))

def vgg16_predict(model, y, sr):
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr:
        return None
    model.eval()
    category_scores = []
    for i in range(sr, len(y), sr // 2):
        S = librosa.feature.melspectrogram(y=y[i - sr: i], sr=sr, n_fft=512, win_length=400, hop_length=160)
        S = librosa.power_to_db(S)
        S = cv2.resize(S, (224, 224))
        S = (S - S.min()) / (S.max() - S.min()) * 255
        S = torch.tensor(S).unsqueeze(0).unsqueeze(0)
        S = S.repeat(1, 3, 1, 1)
        with torch.no_grad():
            y_pred = vgg_model(S)
        category_scores.append(y_pred.numpy()[0][:3])
    
    return ('ar', 'en', 'zh-TW')[np.argmax(sum(category_scores))]

# test vgg16
test_data = AudioDataset(16000)
score = 0
(y, sr), language = test_data[np.random.randint(len(test_data))]

prediction = vgg16_predict(vgg_model, y, sr)
    
print('Prediction: ', prediction)
print('Actual: ', language)