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

import os
import numpy as np
import kaldiio


def is_valid_file_path(path):
    return path and os.path.exists(path)


def load_scp_if_valid(scp_path):
    if is_valid_file_path(scp_path):
        return kaldiio.load_scp(scp_path)
    else:
        if scp_path is not None:
            print('Failed to load scp {}.'.format(scp_path))
        return None


def load_ali_if_valid(ali_path):
    if is_valid_file_path(ali_path):
        # Align files are generally small so we can load them at once.
        # (val - 1) is to convert the phoneme indices into zero-based.
        with kaldiio.ReadHelper(
                'ark: gunzip -c {} |'.format(ali_path)) as reader:
            return {key: (val - 1) for key, val in reader}
    else:
        if ali_path is not None:
            print('Failed to load ali {}.'.format(ali_path))
        return None

def get_ele_if_exist(key, dict_like_obj):
    if dict_like_obj is not None:
        if key in dict_like_obj:
            return dict_like_obj[key]
        else:
            print('Key {} does not exist'.format(key))
            return None
    else:
        print('The dict-like object is empty.')
        return None


class KaldiInterface(object):
    def __init__(self, wav_scp=None, align_path=None, bnf_scp=None,
                 trippg_scp=None, monoppg_scp=None):
        """
        Accessing features like accessing a dictionary.

        Each feature type is stored in a dictionary-like object, and keyed
        by an utterance-id. All feature elements that are keyed by the same
        utterance-id should be computed from the same wavefile. We check this
        assumption in a pythonic way -- we just assume you are a good user
        and we do not check it.

        Base feature types: these are the features that we can load from an
        existing data source. Currently we support,
        - fs and waveform: read from a kaldi scp file.
        - frame alignment: read from a kaldi ali.JOB.gz file.
        - bottleneck features from an AM: read from a kaldi scp file.
        - Triphone PPG from an AM: read from a kaldi scp file.
        - Monophone PPG from an AM: read from a kaldi scp file.

        To add a new **base** feature type "foo",
        - Create a dictionary-like instance variable called "_foo_scp" in the
        **Base feature types** section of the __init__ method.
        - Write a function (try not to create a class method here) that reads
        from your data source and returns a dictionary-like object that is
        indexed by the utterance-id.
        - Initialize "_foo_scp" with your function.
        - Create an instance method "_get_foo(self, utt_id)" that can return a
        feature given an utterance-id. You can apply any transformation you want
        in the method.
        - Register the getter for "foo" in "get_feature"
        - Update this doc with the new base type.

        Derived feature types: these are features that we can compute with the
        existing feature types, e.g., mel-spectra.
        - Cached derived feature types: the cost to store them in the memory is
        low and we want to cache them; to create such types, for example, "bar"
            - Create an instance variable "_bar_scp = {}" in the
            **Cached derived feature types** in the __init__ method.
            - Create an instance method "_get_bar(self, utt_id)" that can return
            this derived feature given an utt-id; store the return value in
            "_bar_scp", see "_get_mel" for an example.
            - Register the getter for "bar" in "get_feature"
        - Other derived feature types: the cost to store them in the memory is
        high or we can easily re-compute them on the fly; to create such types,
        for example, "baz"
            - Create an instance method "_get_baz(self, utt_id)" that can return
            this derived feature given an utt-id
            - Register the getter for "baz" in "get_feature"

        Args:
            wav_scp: wav scp.
            align_path: align .gz file.
            bnf_scp: bottleneck feature scp.
            trippg_scp: triphone ppg scp.
            monoppg_scp: monophone ppg scp.
        """

        # Mel-extraction related. Hard-code the values since my vocoders were
        # trained on these params so there is not need for me to change these.

        # **Base feature types**
        # Read in all scps. wav, bnf, tri-ppg, mono-ppg are "lazy dicts," they
        # only load data when requested.
        self._wav_scp = load_scp_if_valid(wav_scp)
        self._align_scp = load_ali_if_valid(align_path)
        self._bnf_scp = load_scp_if_valid(bnf_scp)
        self._trippg_scp = load_scp_if_valid(trippg_scp)
        self._monoppg_scp = load_scp_if_valid(monoppg_scp)
        # **Base feature types**

        # **Cached derived feature types**
        self._mel_scp = {}
        # **Cached derived feature types**

    def get_feature(self, utt_id: str, feature_type: str):
        """
        Args:
            utt_id: an utterance id.
            feature_type: one of the following. Basically anything you have a
            getter for.
            'fs' | 'wav' | 'fs_and_wav' | 'mel' | 'align' | 'bnf' | 'trippg' |
            'monoppg' | 'mel_align' | 'mel_bnf_align' | 'mel_bnf' |
            'mel_trippg_align' | 'mel_trippg' | 'mel_monoppg_align' |
            'mel_monoppg'

        Returns:
            The feature you requested. None if cannot load.
        """
        val = getattr(self, '_get_{}'.format(feature_type))(utt_id)
        # try:
        #     val = getattr(self, '_get_{}'.format(feature_type))(utt_id)
        # except AttributeError:
        #     print('Feature type {} is not registered.'.format(feature_type))
        #     val = None
        return val



    def _get_bnf(self, utt_id):
        # T*D float32 ndarray.
        return get_ele_if_exist(utt_id, self._bnf_scp).astype(np.float32)

