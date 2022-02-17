#!/usr/bin/env bash

# Extract features using kaldi scripts. Better if run line-by-line.

# Set it to the path of the speaker you want to process.
DATA_ROOT=$1

# Some semi-constant variables.
KALDI_ROOT=/mnt/data1/anurag/kaldi
PRETRAIN_ROOT=/mnt/data1/waris/models/librispeech_am/0013_librispeech_v1
AM_ROOT=$PRETRAIN_ROOT/exp/chain_cleaned/tdnn_1d_sp
IE_ROOT=$PRETRAIN_ROOT/exp/nnet3_cleaned/extractor
LANG_CHAIN_ROOT=$PRETRAIN_ROOT/data/lang_chain
LANG_TGLARGE_ROOT=$PRETRAIN_ROOT/data/lang_test_tglarge
BNF_NODE=prefinal-chain.linear
SCRIPT_ROOT=/home/grads/q/quamer.waris/projects/Accentron/kaldi_scripts

# Call create kaldi files to create all the required kaldi files such as wav.scp
# It also tries to fix the oovs there. It treats each utterance as a separate
# speaker to extract utterance-level ivectors.
python $SCRIPT_ROOT/create_kaldi_files.py $DATA_ROOT

# For the rest of the operations we will do it under kaldi/egs/librispeech/s5
cd $KALDI_ROOT/egs/librispeech/s5 || exit 1;

# Fix kaldi data issues and create spk2utt.
utils/fix_data_dir.sh $DATA_ROOT/kaldi

# Extract MFCCs, 40-dims, the high resolution one.
steps/make_mfcc.sh --nj 8 --cmd run.pl --compress false \
--mfcc-config conf/mfcc_hires.conf $DATA_ROOT/kaldi $DATA_ROOT/kaldi/mfcc/log \
$DATA_ROOT/kaldi/mfcc

# Get utterance-level ivectors.
steps/online/nnet2/extract_ivectors.sh --nj 8 --cmd run.pl --compress false \
$DATA_ROOT/kaldi $LANG_CHAIN_ROOT $IE_ROOT $DATA_ROOT/kaldi/ivector

# # Get alignemnts.
# steps/nnet3/align.sh --nj 8 --cmd run.pl --use_gpu true \
# --online_ivector_dir $DATA_ROOT/kaldi/ivector \
# $DATA_ROOT/kaldi $LANG_TGLARGE_ROOT $AM_ROOT $DATA_ROOT/kaldi/align

# # Combine all ali.gz files into one.
# steps/combine_ali_dirs.sh --nj 1 --combine_lat false --combine_ali true \
# $DATA_ROOT/kaldi $DATA_ROOT/kaldi/align/combined $DATA_ROOT/kaldi/align

# # Convert them from transit ids to phones (in int). Note that output of this
# # step is index from 1 not 0. So when you load the alignments remember to
# # subtract one.
# utils/run.pl $DATA_ROOT/kaldi/align/log/ali_to_phones.1.log \
# $KALDI_ROOT/src/bin/ali-to-phones --per-frame=true $AM_ROOT/final.mdl \
# "ark,t:gunzip -c $DATA_ROOT/kaldi/align/combined/ali.1.gz|" "ark,t:-" \| \
# gzip -c \>$DATA_ROOT/kaldi/align/combined/ali-phone.gz || exit 1;

# # Get alignemnts.
# steps/nnet3/align.sh --nj 8 --cmd run.pl --use_gpu true \
# --online_ivector_dir $DATA_ROOT/kaldi/ivector \
# $DATA_ROOT/kaldi $LANG_TGLARGE_ROOT $AM_ROOT $DATA_ROOT/kaldi/align

# # Combine all ali.gz files into one.
# steps/combine_ali_dirs.sh --nj 1 --combine_lat false --combine_ali true \
# $DATA_ROOT/kaldi $DATA_ROOT/kaldi/align/combined $DATA_ROOT/kaldi/align

# Convert them from transit ids to phones (in int). Note that output of this
# step is index from 1 not 0. So when you load the alignments remember to
# subtract one.
# run.pl $DATA_ROOT/kaldi/align/log/ali_to_phones.1.log \
# ali-to-phones --per-frame=true $AM_ROOT/final.mdl \
# "ark,t:gunzip -c $DATA_ROOT/kaldi/align/combined/ali.1.gz|" "ark,t:-" \| \
# gzip -c \>$DATA_ROOT/kaldi/align/combined/ali-phone.gz || exit 1;


# Make BNF.
steps/nnet3/make_bottleneck_features.sh --cmd run.pl --compress false --nj 8 \
--use_gpu false --ivector-dir $DATA_ROOT/kaldi/ivector $BNF_NODE \
$DATA_ROOT/kaldi $DATA_ROOT/kaldi/bnf $AM_ROOT

# Make triphone PPG. This is compressed into uint16, therefore there may be
# true zero in the data, so remember to add eps when loading the data.
#steps/nnet3/chain/get_phone_post_psi.sh --nj 8 --use_gpu true \
#--online-ivector-dir $DATA_ROOT/kaldi/ivector \
#$AM_ROOT $DATA_ROOT/kaldi $DATA_ROOT/kaldi/trippg

# Make monophone PPG with word postion. We do not compress the output.
#steps/nnet3/chain/get_phone_post_psi.sh --nj 8 --use_gpu true \
#--online-ivector-dir $DATA_ROOT/kaldi/ivector \
#--trans-mat-path $PRETRAIN_ROOT/data/transform_with_word_position.mat \
#$AM_ROOT $DATA_ROOT/kaldi $DATA_ROOT/kaldi/monoppg-word-position
