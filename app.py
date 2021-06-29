from flask import Flask
from flask import request  # for methods
import os

import python_speech_features
import numpy as np
from tensorflow import keras
import librosa
from flask_cors import CORS 
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

cors_config = {
  "origins": ["http://localhost:3000"],
  "methods": ["OPTIONS", "GET", "POST"],
  "allow_headers": '*'
}
CORS(app,resources={r"/stt": cors_config}) 

SAMPLING_RATE = 8000
word2idx = {
    'right': 0,
    'on': 1,
    'off': 2,
    'stop': 3,
    'up': 4,
    'yes': 5,
    'down': 6,
    'left': 7,
    'no': 8,
    'go': 9,
    'backward': 10,
}


idx2word = [word for word in word2idx]

loaded_model = keras.models.load_model(os.path.join('model', '3-convs_95acc.h5'))
# loaded_model = keras.models.load_model(os.path.join('/home/MtgTemp/mysite/model', '3-convs_95acc.h5'))
 

def stt(file,sr):
    features = wav2modelInput(file, sr)
    return predict(features)


def wav2modelInput(wav, sr):
    samples = librosa.resample(wav, sr, SAMPLING_RATE)
    samples = extract_loudest_section(samples, SAMPLING_RATE)
    if len(samples) > SAMPLING_RATE:
        samples = samples[:SAMPLING_RATE]
    else:
        samples = np.pad(
            samples, (0, max(0, SAMPLING_RATE - len(samples))), "constant")

    return samples2feature(samples)


def extract_loudest_section(audio, length):
    audio = audio.astype(np.float)  # to avoid integer overflow when squaring
    audio_pw = audio**2  # power
    window = np.ones((length, ))
    conv = np.convolve(audio_pw, window, mode="valid")
    begin_index = conv.argmax()
    return audio[begin_index:begin_index+length]


def samples2feature(data):
    data = data.astype(np.float)
    # normalize data
    data -= data.mean()
    data /= np.max((data.max(), -data.min()))
    # compute MFCC coefficients
    features = python_speech_features.mfcc(data, samplerate=SAMPLING_RATE, winlen=0.025, winstep=0.01, numcep=20,
                                           nfilt=40, nfft=512, lowfreq=100, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
    return features


def predict(input):
    input_shape = (99, 20)
    prob = loaded_model.predict(np.reshape(
        input, (1, input_shape[0], input_shape[1])))
    index = np.argmax(prob[0])
    if(prob[0][index]>0.80):
    	return idx2word[index]
    else:
    	return ""




@app.route("/stt", methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        file = request.files['audio']
        samples, sr=librosa.load(file)
        res=stt(samples, sr)
        return { "result": res }    
    else:
        return {
            "error": 'Not post request!'
        }
 
if __name__ == "__main__":
    app.run(debug=True)
