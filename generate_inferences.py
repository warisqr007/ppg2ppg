from synthesizer.inference import Synthesizer
from synthesizer_like_translator.inference import Synthesizer as Translator
from synthesizer.kaldi_interface import KaldiInterface
from encoder import inference as encoder
from encoder import inference_accent as encoder_accent
from vocoder import inference as vocoder
import numpy as np
from pathlib import Path
from utils.argutils import print_args
import random
import librosa
import IPython.display as ipd
from synthesizer.hparams import hparams
import os
from scipy.io import wavfile
from tqdm import tqdm


def synthesize(bnf, embed):
    spec = synthesizer.synthesize_spectrograms([bnf], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav

def translate_ppg(bnf, embed):
    spec = translator.synthesize_spectrograms([bnf], [embed])[0]
    return spec


def generate_accent_embed(src_utterance_path):
    wav, _ = librosa.load(src_utterance_path, hparams.sample_rate)
    wav = encoder.preprocess_wav(wav)
    embed_accent = encoder_accent.embed_utterance(wav)

    return embed_accent

def generate_speaker_embed(tgt_utterance_path):
    wav, _ = librosa.load(tgt_utterance_path, hparams.sample_rate)
    wav = encoder.preprocess_wav(wav)
    embed_speaker = encoder.embed_utterance(wav)

    return embed_speaker


encoder_speaker_weights = Path("/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/encoder/saved_models/pretrained.pt")
encoder_accent_weights = Path("/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/encoder/saved_models/encoder_accent.pt")
vocoder_weights = Path("/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("/mnt/data1/waris/model_outputs/accentron/parallel/logs-Accetron_train_parallel/taco_pretrained")
syn_dir_trans = Path("/mnt/data1/waris/model_outputs/translator/sythesizer_like_train_set/logs-translator_train/taco_pretrained")

encoder.load_model(encoder_speaker_weights)
encoder_accent.load_model(encoder_accent_weights)
synthesizer = Synthesizer(syn_dir)
translator = Translator(syn_dir_trans)
vocoder.load_model(vocoder_weights)

utterance_ids = ['arctic_b0'+str(i) for i in range(490, 540)]
speakers = ['BDL', 'NJS', 'TXHC', 'YKWK', 'ZHAA']

for speaker in tqdm(speakers, desc="Speakers ", position=0):
    for utterance_id in tqdm(utterance_ids, desc="Utterances per speaker", position=1, leave=False):
        kaldi_dir = '/mnt/data1/waris/datasets/data/arctic_dataset/all_data/'+speaker+'/kaldi'
        ki = KaldiInterface(wav_scp=str(os.path.join(kaldi_dir, 'wav.scp')),
                            bnf_scp=str(os.path.join(kaldi_dir, 'bnf/feats.scp')))
        bnf = ki.get_feature('_'.join([speaker, utterance_id]), 'bnf')

        acc_utterance_path = '/mnt/data1/waris/datasets/data/arctic_dataset/all_data/BDL/wav/'+str(utterance_id)+'.wav'
        embed_accent = generate_accent_embed(acc_utterance_path)
        bnf_native = translate_ppg(bnf, embed_accent)

        tgt_utterance_path = '/mnt/data1/waris/datasets/data/arctic_dataset/all_data/'+str(speaker)+'/wav/'+str(utterance_id)+'.wav'
        embed_speaker = generate_speaker_embed(tgt_utterance_path)

        synthesis_wav = synthesize(bnf_native, embed_speaker)

        output_dir = '/mnt/data1/waris/model_results/translator_synthesizer/ppg_to_ppg_294k/'+str(speaker)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = str(utterance_id)+'.wav'
        output_file = os.path.join(output_dir, filename)
        wavfile.write(output_file, hparams.sample_rate, synthesis_wav)
        # wavfile.write(output_file, hparams.sample_rate, synthesis_wav.astype(np.int16))

print("Done generating inferences!")