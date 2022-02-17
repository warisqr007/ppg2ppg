from data_objects.preprocess import create_dvec_embeddings_test
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the test.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("target_speaker", type=str, help= \
        "Target speaker. The converted sample would contain the voice identity of this speaker.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default="/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/encoder/saved_models/pretrained.pt", help= \
                            "Path your trained encoder model.")
    parser.add_argument("-ea", "--encoder_accent_model_fpath", type=Path,
                        default="/home/grads/q/quamer.waris/projects/Accentron/pretrained_model/pretrained/encoder/saved_models/resnet_encoder_accent/encoder_accent.pth", help= \
                            "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    parser.add_argument("-emb", "--embedding_type", type=str, default="speaker", help= \
        "Type of embedding to generate"
        "options: both, speaker, accent")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    create_dvec_embeddings_test(**vars(args))
