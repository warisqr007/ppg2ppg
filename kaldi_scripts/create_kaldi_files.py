# Copyright 2020 Guanlong Zhao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import sys

"""
Read in files in folders like,
/L2-ARCTIC/bdl
and create kaldi-required files. wav.scp etc. Need to call utils/fix_data_dir.sh
afterwards to fix the files for kaldi conventions.
"""


def fix_oov(line: str):
    """
    Args:
        line: transcript like "WORD1 WORD2 WORD3..." separated by space.

    Returns:
        OOV replaced.
    """
    oov_dict = {
        "'EM": "EM",
        "MCVEIGH": "MC VEIGH",
        "MCFEE'S": "MC FEE'S",
        "DENNIN'S": "DENNING'S",
        "DAUGHTRY'S": "DAW TREE'S",
        "DAUGHTRY": "DAW TREE"
    }
    line_split = line.split()
    for idx, word in enumerate(line_split):
        if word in oov_dict:
            print('Changing OOV {} to {}'.format(word, oov_dict[word]))
            line_split[idx] = oov_dict[word]
    return ' '.join(line_split)


if __name__ == '__main__':
    cache_folder = sys.argv[1]
    output_dir = os.path.join(cache_folder, 'kaldi')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    speaker_id = os.path.basename(cache_folder)
    wav_file_paths = glob.glob(os.path.join(cache_folder, 'wav/*.wav'))
    wav_file_paths.sort()

    wav_scp = open(os.path.join(output_dir, 'wav.scp'), 'w')
    utt2spk = open(os.path.join(output_dir, 'utt2spk'), 'w')
    text = open(os.path.join(output_dir, 'text'), 'w')

    # Last line should not include the newline break
    for ii, each_wav in enumerate(wav_file_paths):
        line_break = '\n'
        if ii + 1 == len(wav_file_paths):
            line_break = ''

        wav_file_name = os.path.basename(each_wav)
        wav_file_id = wav_file_name.split('.')[0]
        utt_id = '{}_{}'.format(speaker_id, wav_file_id)
        wav_scp.write('{} {}{}'.format(utt_id, each_wav, line_break))
        utt2spk.write('{} {}{}'.format(utt_id, utt_id, line_break))

        # # lab files share the same name as the wav files
        # lab_file_path = os.path.join(cache_folder, 'text',
        #                              '{}.lab'.format(wav_file_id))
        # with open(lab_file_path, 'r') as f:
        #     transcript = fix_oov(f.read().splitlines()[0].upper())
        #     text.write('{} {}{}'.format(utt_id, transcript, line_break))
    wav_scp.close()
    utt2spk.close()
    text.close()

