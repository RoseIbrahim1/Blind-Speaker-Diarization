import os
import shutil
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from spectralcluster import SpectralClusterer, RefinementOptions

def del_sub_dir(pathsub, dirname):
    folder = os.path.join(pathsub, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)
        return

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def create_labelling(labels, wav_splits):
    sampling_rate = 16000
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i - 1]:
            labelling.append((str(labels[i - 1]), start_time, time))
            start_time = time
        if i == len(times) - 1:
            labelling.append((str(labels[i]), start_time, time))

    return labelling

def diar(fpath, spk_num):
    audio_file_path = fpath
    wav_fpath = Path(audio_file_path)

    wav = preprocess_wav(wav_fpath)
    if len(wav) == 0:
        return [], []

    encoder = VoiceEncoder("cpu")
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=16, min_coverage=0.75
    )

    refinement = RefinementOptions(
        gaussian_blur_sigma=1,
        p_percentile=0.5
    )

    clusterer = SpectralClusterer(
        min_clusters=spk_num,
        max_clusters=spk_num,
        refinement_options=refinement
    )

    labels = clusterer.predict(cont_embeds)
    labelling = create_labelling(labels, wav_splits)

    return labelling, wav
