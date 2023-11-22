import waveglow
import text
import audio
import utils
import torch
import argparse
import numpy as np
import os
from tqdm import tqdm

def synthesis(model, phn, alpha=1.0, train_config=None):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(train_config):
    tests = [ 
        "I am very happy to see you again!",
        "Durian model is a very good speech synthesis!",
        "When I was twenty, I fell in love with a girl.",
        "I remove attention module in decoder and use average pooling to implement predicting r frames at once",
        "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted.",
        "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list



def main(model, results):
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    model.load_state_dict(torch.load(model, map_location='cuda:0')['model'])
    model = model.eval()


    data_list = get_data()
    for speed in [0.8, 1., 1.3]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(model, phn, speed)
            
            os.makedirs(results, exist_ok=True)
            
            audio.tools.inv_mel_spec(
                mel, f"{results}/s={speed}_{i}.wav"
            )
            
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"{results}/s={speed}_{i}_waveglow.wav"
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('results', type=str)
    args = parser.parse_args()
    main(args.model, args.results)