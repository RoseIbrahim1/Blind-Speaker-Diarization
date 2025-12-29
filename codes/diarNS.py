import os
import shutil
import soundfile as sf
import numpy as np
from diarization import diar, del_sub_dir


def run_diarization(spk_num, file_path):
    rootdir = os.path.dirname(file_path)  # directory containing the file
    sampling_rate = 16000
    seglen = 0
   
    # Clear output folders
    delconca = os.path.join(rootdir, 'concanated')
    if os.path.exists(delconca):
        shutil.rmtree(delconca)
    os.makedirs(delconca, exist_ok=True)

    delsepa = os.path.join(rootdir, 'separated')
    if os.path.exists(delsepa):
        shutil.rmtree(delsepa)
    os.makedirs(delsepa, exist_ok=True)

    # Process the selected file only
    labels, wavf = diar(file_path, spk_num)
    sf.write(os.path.join(rootdir, 'outputNoSilence.wav'), wavf, sampling_rate, 'PCM_24')

    del_sub_dir(rootdir, 'concanated')
    del_sub_dir(rootdir, 'separated')

    speakers = {f'spk{i}': np.array([]) for i in range(spk_num)}

    for label in labels:
        spk_id, start, end = label
        if (end - start) > seglen:
            segment = wavf[int(start * sampling_rate):int(end * sampling_rate)]
            speaker_key = f'spk{spk_id}'
            sepa_path = os.path.join(rootdir, 'separated', f'{speaker_key}_{start}.wav')
            sf.write(sepa_path, segment, sampling_rate, 'PCM_24')
            speakers[speaker_key] = np.concatenate((speakers[speaker_key], segment), axis=0)
            print(f"{speaker_key.upper()} catched...")

    for spk_id, data in speakers.items():
        conca_path = os.path.join(rootdir, 'concanated', f'{spk_id}.wav')
        sf.write(conca_path, data, sampling_rate, 'PCM_24')

