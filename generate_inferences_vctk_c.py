from synthesizer.inference import Synthesizer
from synthesizer.kaldi_interface import KaldiInterface
from encoder import inference as encoder
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
import shutil


def synthesize(bnf, embed):
    spec = synthesizer.synthesize_spectrograms([bnf], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav

def generate_speaker_embed(tgt_utterance_path):
    wav, _ = librosa.load(tgt_utterance_path, hparams.sample_rate)
    wav = encoder.preprocess_wav(wav)
    embed_speaker = encoder.embed_utterance(wav)

    return embed_speaker


encoder_speaker_weights = Path("/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/encoder/saved_models/pretrained.pt")
vocoder_weights = Path("/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("/mnt/data1/waris/model_outputs/accentron/parallel_vctk_mic2/logs-Accetron_train_parallel_vctk/taco_pretrained")

encoder.load_model(encoder_speaker_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

# speaker_utterance_seen = {}
# speaker_utterance_unseen = {}

# speakers = ['p253',
#             'p241',
#             'p261',
#             'p308',
#             'p251',
#             'p273',
#             'p279',
#             'p246',
#             'p283',
#             'p362',
#             'p376',
#             'p260',
#             'p282',
#             'p229',
#             'p276',
#             'p271',
#             'p269',
#             'p252',
#             'p300',
#             'p267',
#             'p304',
#             's5']

# for speaker in speakers:
#     speaker_utterance_seen[speaker] = []
#     speaker_utterance_unseen[speaker] = []

# src_speaker = 'p250'
# src_speaker_utterances={}
# src_speaker_utterances['seen'] = []
# src_speaker_utterances['unseen'] = []

# src_path = '/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/'+src_speaker+'/wav/'
# _, _, utterance_ids = next(os.walk(src_path))

# src_utterances_dir ='/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/source_utterances/seen/'
# count = 0
# for utterance_id in utterance_ids:
#     if count >=100:
#         continue

#     if  "_mic2.wav" in utterance_id:
#         count = count+1

#         src_speaker_utterances['seen'].append(utterance_id)

#         if not os.path.isdir(src_utterances_dir):
#             os.makedirs(src_utterances_dir)
#         shutil.copy(src_path+utterance_id, src_utterances_dir)
#         for speaker in speakers:
#             kaldi_dir = '/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/'+src_speaker+'/kaldi'
#             ki = KaldiInterface(wav_scp=str(os.path.join(kaldi_dir, 'wav.scp')),
#                                 bnf_scp=str(os.path.join(kaldi_dir, 'bnf/feats.scp')))
#             bnf = ki.get_feature('_'.join([src_speaker, utterance_id[:-4]]), 'bnf')

#             tgt_utterance_path = '/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/'+str(speaker)+'/wav/'+str(speaker)+str(utterance_id)[4:]
#             if not os.path.exists(tgt_utterance_path):
#                 continue
            
#             target_dir = '/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/target_utterances/'+str(speaker)+'/seen/'
#             if not os.path.isdir(target_dir):
#                 os.makedirs(target_dir)
#             shutil.copy(tgt_utterance_path, target_dir)
#             speaker_utterance_seen[speaker].append(str(speaker)+str(utterance_id)[4:])
#             embed_speaker = generate_speaker_embed(tgt_utterance_path)

#             synthesis_wav = synthesize(bnf, embed_speaker)

#             output_dir = '/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/converted_utterances/'+str(speaker)
#             if not os.path.isdir(output_dir):
#                 os.makedirs(output_dir)
#             filename = str(utterance_id)[5:]
#             output_file = os.path.join(output_dir, filename)
#             wavfile.write(output_file, hparams.sample_rate, synthesis_wav)

# print("Done generating one-to-many inferences!")

# src_utterances_dir ='/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/source_utterances/unseen/'
# count = 0
# for utterance_id in utterance_ids:
#     if count >=100:
#         continue

#     if  "_mic2.wav" in utterance_id and utterance_id not in src_speaker_utterances['seen']:
#         count = count+1

#         src_speaker_utterances['unseen'].append(utterance_id)

#         if not os.path.isdir(src_utterances_dir):
#             os.makedirs(src_utterances_dir)
#         shutil.copy(src_path+utterance_id, src_utterances_dir)

#         for speaker in speakers:
#             tgt_utterance_path = '/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/'+str(speaker)+'/wav/'+str(speaker)+str(utterance_id)[4:]
#             if not os.path.exists(tgt_utterance_path):
#                 continue
            
#             target_dir = '/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/target_utterances/'+str(speaker)+'/unseen/'
#             if not os.path.isdir(target_dir):
#                 os.makedirs(target_dir)
#             shutil.copy(tgt_utterance_path, target_dir)
#             speaker_utterance_unseen[speaker].append(str(speaker)+str(utterance_id)[4:])

# print("Done Copying unseen samples!")

source_speakers = ['p253',
            'p241',
            'p261',
            'p308',
            'p251',
            'p273',
            'p279',
            'p246',
            'p283',
            'p362',
            'p376',
            'p260',
            'p282',
            'p229',
            'p276',
            'p271',
            'p269',
            'p252',
            'p300',
            'p267',
            'p304',
            's5']


target_speaker = 'p250'

print("Start generating many-to-many inferences!")

for speaker in source_speakers:
    print(speaker)
    speaker_fpath_seen = "/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/target_utterances/"+speaker+"/seen"
    _, _, utterance_ids_seen = next(os.walk(speaker_fpath_seen))
    utterance_ids = []
    utterance_ids.extend(utterance_ids_seen)
    speaker_fpath_unseen = "/mnt/data1/waris/model_results/vc_vctk_all/one_to_many/target_utterances/"+speaker+"/unseen"
    _, _, utterance_ids_unseen = next(os.walk(speaker_fpath_unseen))
    utterance_ids.extend(utterance_ids_unseen)

    for utterance in utterance_ids:
        kaldi_dir = '/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/'+speaker+'/kaldi'
        ki = KaldiInterface(wav_scp=str(os.path.join(kaldi_dir, 'wav.scp')),
                            bnf_scp=str(os.path.join(kaldi_dir, 'bnf/feats.scp')))
        bnf = ki.get_feature('_'.join([speaker, utterance[:-4]]), 'bnf')

        tgt_utterance_path = '/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/'+str(target_speaker)+'/wav/'+str(target_speaker)+str(utterance)[4:]
        
        embed_speaker = generate_speaker_embed(tgt_utterance_path)

        synthesis_wav = synthesize(bnf, embed_speaker)

        output_dir = '/mnt/data1/waris/model_results/vc_vctk_all/many_to_many/converted_utterances/'+str(speaker)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = str(utterance)[5:]
        output_file = os.path.join(output_dir, filename)
        wavfile.write(output_file, hparams.sample_rate, synthesis_wav)

print("Done generating many-to-many inferences!")