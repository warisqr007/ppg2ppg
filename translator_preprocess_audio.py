from data_objects.preprocess_translator import preprocess_l2arctic
from config.hparams import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_root", type=Path, help=\
        "Path to L2-ARCTIC dataset.")
    parser.add_argument("-ts", "--test_speakers", type=str, nargs="*", default=[], help= \
        "Test speakers. These speakers will be excluded from the training.")
    parser.add_argument("-ref", "--reference_speakers", type=str, nargs="*", help= \
        "Reference speakers. This should correspond to the KALDI speakers")
    parser.add_argument("-kaldi", "--kaldi_dirs", type=Path, nargs="*", help= \
        "Paths to kaldi dir for reference speakers")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/translator/")
    parser.add_argument("-n", "--n_processes", type=int, default=16, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    args = parser.parse_args()
    
    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.dataset_root.joinpath("SV2TTS", "translator")

    # Create directories
    assert args.dataset_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)
    preprocess_l2arctic(**vars(args))


#Example: python translator_preprocess_audio.py /mnt/data1/waris/datasets/data_ppg2ppg_v2 -ts NJS YKWK ZHAA TXHC LXHC SV2TTS -ref BDL SVBI -kaldi /mnt/data1/waris/datasets/data_ppg2ppg_v2/BDL/kaldi /mnt/data1/waris/datasets/data_ppg2ppg_v2/SVBI/kaldi