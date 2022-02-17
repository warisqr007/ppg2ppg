from pickle import NONE
from synthesizer.inference import Synthesizer
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
import argparse
#from functools import partial
#from multiprocessing.pool import Pool


def synthesize(synthesizer, bnf, embed):
    spec = synthesizer.synthesize_spectrograms([bnf], [embed])[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav

def generate_inference(fpath, synthesizer_root: Path, synthesizer, output_dir: Path):

    ppg_path, embed_path = fpath
    
    bnf = np.load(ppg_path)
    embed_speaker = np.load(embed_path)

    synthesis_wav = synthesize(synthesizer, bnf, embed_speaker)
    speaker_name = (ppg_path.name).split('-')[1]
    utterance_id = (ppg_path.name).split('-')[-1].split("_")[1]
    output_fpath = synthesizer_root.joinpath("test_wav", str(speaker_name))    #'/mnt/data1/waris/model_results/translator_synthesizer/ppg_to_ppg_200k/'+str(speaker)
    if output_dir is not None:
        output_fpath = output_dir.joinpath("converted_wav", str(speaker_name)) 
    if not os.path.isdir(output_fpath):
        os.makedirs(output_fpath)
    filename = str(utterance_id)+'.wav'
    output_file = output_fpath.joinpath(filename)
    wavfile.write(output_file, hparams.sample_rate, synthesis_wav)

def generate_inferences_vctk(synthesizer_root: Path, output_dir: Path, synthesizer_models_fpath: Path, vocoder_model_fpath: Path, n_processes: int):

    ppg_dir = synthesizer_root.joinpath("ppgs")
    embed_dir = synthesizer_root.joinpath("embeds")
    metadata_fpath = synthesizer_root.joinpath("test.txt")
    assert ppg_dir.exists() and metadata_fpath.exists() and embed_dir.exists()

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(ppg_dir.joinpath(m[2]), embed_dir.joinpath(m[3])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    #func = partial(generate_inference, synthesizer_models_fpath=synthesizer_models_fpath, vocoder_model_fpath=vocoder_model_fpath)
    #job = Pool(n_processes).imap(func, fpaths)

    if not vocoder.is_loaded():
        vocoder.load_model(vocoder_model_fpath)
    
    synthesizer = Synthesizer(synthesizer_models_fpath)

    for fpath in tqdm(fpaths, "Generating Test Inferenes", len(fpaths), unit="utterances"):
        generate_inference(fpath, synthesizer_root, synthesizer, output_dir)
    #list(tqdm(job, "Generating Test Inferenes", len(fpaths), unit="utterances"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer testing data that contains the audios and the text.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-out", "--output_dir", type=Path,
                        default="/mnt/data1/waris/model_results/vc_vctk_all/many_to_many", help= \
                            "Path your trained Synthesizer model.")
    parser.add_argument("-s", "--synthesizer_models_fpath", type=Path,
                        default="/mnt/data1/waris/model_outputs/accentron/parallel_vctk/logs-Accetron_train_parallel_vctk/taco_pretrained", help= \
                            "Path your trained Synthesizer model.")
    parser.add_argument("-v", "--vocoder_model_fpath", type=Path,
                        default="/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/vocoder/saved_models/pretrained/pretrained.pt", help= \
                            "Path your trained vocoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    generate_inferences_vctk(**vars(args))