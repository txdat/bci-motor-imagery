import glob
import random

import mne
import pandas as pd


def gen_subject_raw(subject_dir, shuffle=False):
    list_of_raw = list()
    for edf_file in sorted(glob.glob(f"{subject_dir}/sample*/EEG.edf")):
        raw: mne.io.Raw = mne.io.read_raw_edf(edf_file, verbose=False)
        list_of_raw.append(raw)

    if shuffle:
        random.shuffle(list_of_raw)

    raw = mne.concatenate_raws(list_of_raw)

    mne.export.export_raw(
        f"{subject_dir}/experimentEEG.edf", raw, fmt="edf", verbose=False
    )


if __name__ == "__main__":
    gen_subject_raw("../data/vin/Official/K309")
