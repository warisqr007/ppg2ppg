from data_objects.preprocess_translator import create_dvec_embeddings
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/translator/.")
    parser.add_argument("-ea", "--encoder_accent_model_fpath", type=Path,
                        default="/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/encoder/saved_models/encoder_accent.pt", help= \
                            "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    create_dvec_embeddings(**vars(args))

# Example: python translator_preprocess_embeds.py /mnt/data1/waris/datasets/data_ppg2ppg_v2/SV2TTS/translator
