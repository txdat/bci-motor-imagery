# VIN-HMI EEG dataset
# each (action/intention) sample/run belongs to 1 scenario and has 3 or 4 event types
#
# events = {
#     "Resting",
#     "Thinking" (eye closed),
#     "Thinking and Acting",
#     "Typing",
# }
#
# action run:
#     scenarios = {
#         "nâng tay trái",
#         "nâng tay phải",
#         "nâng chân trái",
#         "nâng chân phải",
#         "gật đầu",
#         "lắc đầu",
#         "há miệng",
#     }
#
#     0-------X-------|Thinking|-------X--------|Resting|-------X--------|Thinking and Acting|------X------|Resting|--
#     --------X-------|Thinking|-------X--------|Resting|-------X--------|Thinking and Acting|------X------|Resting|--
#     --------X-------|Thinking|-------X--------|Resting|-------X--------|Thinking and Acting|------X------|Resting|--
#     --------X-------|Typing  |-------X-->>
#
# intention run:
#     scenarios = {
#         "tôi muốn uống nước",
#         "tôi muốn vệ sinh",
#     }
#
#     0-------X-------|Thinking|-------X--------|Resting|---------
#     --------X-------|Thinking|-------X--------|Resting|---------
#     --------X-------|Thinking|-------X--------|Resting|---------
#     --------X-------|Typing  |-------X-->>

import glob
import json
import logging
import os
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
from scipy import linalg, stats
import pandas as pd
import mne
from braindecode.datasets import (
    BaseDataset,
    BaseConcatDataset,
    create_from_mne_epochs,
)
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from tqdm import tqdm

logging.getLogger("mne").setLevel(logging.ERROR)

# fmt: off
ACTION_SCENARIOS = {
    "nâng tay trái",
    "nâng tay phải",
    "nâng chân trái",
    "nâng chân phải",
    "gật đầu",
    "lắc đầu",
    "há miệng",
}

INTENTION_SCENARIOS = {
    "tôi muốn uống nước",
    "tôi muốn vệ sinh",
}

DEFAULT_SCENARIOS = ACTION_SCENARIOS | INTENTION_SCENARIOS

DEFAULT_EVENTS = {
    "Resting",
    "Thinking",
    "Thinking and Acting",
    "Typing",
}

DEFAULT_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FT9", "FC5", "FC1", "FC2", "FC6", "FT10",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]
# fmt: on


def load_subjects(data_dir: str) -> List[Tuple[str, List[str]]]:
    """
    load subjects and their scenarios
    """

    subjects = list()
    for d in glob.glob(f"{data_dir}/*"):
        if os.path.exists(f"{d}/info.json"):
            scenarios = list()
            for s in glob.glob(f"{d}/sample*"):
                if (
                    not os.path.isdir(s)
                    or not os.path.exists(f"{s}/EEG.edf")
                    or not os.path.exists(f"{s}/eeg.json")
                ):
                    continue

                with open(f"{s}/eeg.json", mode="r") as f:
                    scenario = json.load(f).get("Scenario", "").strip().lower()
                    scenarios.append(scenario)

            subjects.append((d.split("/")[-1], scenarios))

        elif os.path.isdir(d) and not os.path.exists(f"{d}/EEG.edf"):
            subjects.extend(load_subjects(d))

    return subjects


def preprocess_data(
    ds: BaseConcatDataset,
    channels: Optional[List[str]] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    moving_standardize: bool = False,
) -> None:
    """
    preprocess raw data inplace
    """

    def convert_volt_to_micro_volt(x):
        return x * 1e6

    # init preprocessor
    # processor's function is mne raw's method
    # https://braindecode.org/auto_examples/plot_bcic_iv_2a_moabb_cropped.html#loading-and-preprocessing-the-dataset
    preprocessors = [
        # Preprocessor("pick_types", eeg=True, stim=False),  # Keep EEG sensors
        Preprocessor(
            "pick_channels",
            ch_names=channels or DEFAULT_CHANNELS,
            ordered=True,
        ),
    ]
    if fmin is not None or fmax is not None:
        preprocessors.append(
            Preprocessor(
                "filter",
                l_freq=fmin,
                h_freq=fmax,
                n_jobs=-1,
                verbose=False,
            )  # Bandpass filter using fir
        )
    if moving_standardize:
        preprocessors.append(
            Preprocessor(convert_volt_to_micro_volt)
        )  # Convert from V to uV
        preprocessors.append(
            Preprocessor(
                exponential_moving_standardize,
                factor_new=1e-3,
                init_block_size=None,
            )  # Exponential moving standardization
        )

    # do preprocessing inplace
    preprocess(ds, preprocessors)


def load_sample_data(
    sample_dir: str,
    scenarios: Optional[List[str]] = None,
    extra_description: Optional[Dict[str, Any]] = None,
) -> Optional[BaseDataset]:
    """
    load sample's raw data
    """

    scenarios = set(scenarios or DEFAULT_SCENARIOS)

    if (
        not os.path.isdir(sample_dir)
        or not os.path.exists(f"{sample_dir}/EEG.edf")
        or not os.path.exists(f"{sample_dir}/eeg.json")
    ):
        return None

    with open(f"{sample_dir}/eeg.json", mode="r") as f:
        scenario = json.load(f).get("Scenario", "").strip().lower()

        # fix scenario's name
        if scenario == "nang tay trai":
            scenario = "nâng tay trái"

        assert (
            scenario in DEFAULT_SCENARIOS
        ), f"{sample_dir} data has invalid scenario {scenario}"

    if len(scenarios) > 0 and scenario not in scenarios:
        return None

    raw = mne.io.read_raw_edf(f"{sample_dir}/EEG.edf", preload=False, verbose=False)
    raw.set_montage("standard_1005")

    # fix event's name

    assert all(
        event in DEFAULT_EVENTS for event in raw.annotations.description
    ), f"{sample_dir} data has invalid events {raw.annotations.description}"

    # TODO: fix missing data with ET, EEGTimeStamp data

    # fix wrong annotation in some subjects
    if scenario in INTENTION_SCENARIOS:
        raw.annotations.description[
            np.where(raw.annotations.description == "Thinking and Acting")[0]
        ] = "Thinking"

    description = {"scenario": scenario}
    if extra_description is not None:
        description.update(extra_description)

    return BaseDataset(raw, description=description)


def load_subject_data(
    subject_dir: str,
    scenarios: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    channels: Optional[List[str]] = None,
    label_mapping: Optional[Dict[str, str]] = None,
    minimal_trial_duration: float = 1.0,
    window_duration: float = 1.0,
    window_stride_duration: float = 0.25,
    start_offset: float = 0,
    stop_offset: float = 0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    moving_standardize: bool = False,
    resample: Optional[float] = None,
    return_raw: bool = False,
    return_preprocessed: bool = False,
    preload: bool = False,
) -> BaseConcatDataset:
    """
    load windowed variable-length trials from Official VIN dataset for target subject

    **Note**
        all "Resting" trials of all subject's runs are considered to be similar

    // all in seconds
                                                   trial onset +
                           trial onset          trial duration
       ----|--------------------|------------------------|-----------------------|----
           trial onset +                                             trial onset +
    trial start_offset                                            trial duration +
                                                               trial stop_offset

    Return
    ----------------------------
    ds      concat dataset
        concatenated dataset whose subdataset belongs to subject's trial (raw's epochs)

        ds.description      pd dataframe
            scenario
            event
            trial_idx   index of trial on run (sample)
            split       split tag for braindecode's dataset splitting, in ["train", "valid", "test"]
    """

    subject = subject_dir.split("/")[-1]

    # load raw datasets
    # each dataset belongs to sample
    list_of_ds = list()
    for sample_dir in tqdm(sorted(glob.glob(f"{subject_dir}/sample*")), desc="sample"):
        sample_ds = load_sample_data(sample_dir, scenarios)
        if sample_ds is not None:
            list_of_ds.append(sample_ds)

    ds = BaseConcatDataset(list_of_ds)

    sfreq = ds.datasets[0].raw.info["sfreq"]
    # assert all(
    #     sample_ds.raw.info["sfreq"] == sfreq for sample_ds in ds.datasets
    # ), f"{subject_dir}'s samples have different sampling frequencies"
    for sample_ds in ds.datasets:
        sample_sfreq = sample_ds.raw.info["sfreq"]
        assert sample_sfreq == sfreq, (
            f"{subject}'s {sample_ds.description['scenario']} sample "
            f"doesn't have matched sampling frequency ({sample_sfreq}Hz, not {sfreq}Hz)"
        )

    if return_raw:
        return ds

    preprocess_data(ds, channels, fmin, fmax, moving_standardize)

    if return_preprocessed:
        return ds

    # create windows dataset
    mapping = {event: _id for _id, event in enumerate(events or DEFAULT_EVENTS)}
    label_mapping = label_mapping or dict()
    label_id = (
        {label_mapping.get("Resting", "Resting"): 0} if "Resting" in mapping else dict()
    )
    for scenario in sorted(ds.description["scenario"].unique()):
        for event in sorted(mapping.keys()):
            if event == "Resting":
                continue
            if scenario in INTENTION_SCENARIOS and event == "Thinking and Acting":
                continue

            label = f"{scenario}_{event}"
            label = label_mapping.get(label, label)
            if label in label_id:
                continue

            label_id[label] = len(label_id)

    # split trials from samples before cropping to windows
    list_of_epochs = list()
    epoch_info = list()
    # minimal_trial_size = int(sfreq * minimal_trial_duration)
    for sample_ds in tqdm(ds.datasets, desc="sample"):
        sample_annot = pd.DataFrame(sample_ds.raw.annotations)
        sample_events, sample_event_id = mne.events_from_annotations(
            sample_ds.raw, verbose=False
        )
        assert len(sample_annot) == len(sample_events)  # load all sample's events

        trial_count = dict()  # trial's index for each default label

        # due to variable-length trials, cannot load all epochs at the same time
        for i, r in sample_annot.iterrows():
            if r["description"] not in mapping:
                continue

            if r["duration"] + stop_offset - start_offset < minimal_trial_duration:
                continue

            # single epoch (1 event only)
            # tmin, tmax are relative to trial's onset (seconds)
            epochs = mne.Epochs(
                sample_ds.raw,
                sample_events[i : i + 1],
                event_id=sample_event_id,
                tmin=start_offset,
                tmax=r["duration"] + stop_offset,
                baseline=None,
                preload=False,
                proj=False,
                on_missing="ignore",
                event_repeated="drop",
                verbose=False,
            )

            # if len(epochs.times) < minimal_trial_size:  # not enough data
            #     continue

            default_label = (
                f"{sample_ds.description['scenario']}_{r['description']}"
                if r["description"] != "Resting"
                else "Resting"
            )
            label = label_mapping.get(default_label, default_label)

            if default_label not in trial_count:
                trial_count[default_label] = 0

            list_of_epochs.append(epochs)
            epoch_info.append(
                (
                    sample_ds.description["scenario"],
                    r["description"],
                    label,
                    label_id[label],
                    trial_count[default_label],
                )
            )

            trial_count[default_label] += 1

    # split each epoch to overlapping sub-epochs (windows)
    ds = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=int(sfreq * window_duration),
        window_stride_samples=int(sfreq * window_stride_duration),
        drop_last_window=True,
    )

    # update trial's data and events
    if resample is not None:
        preload = True

    for i, trial_ds in tqdm(
        enumerate(ds.datasets), total=len(ds.datasets), desc="trial"
    ):
        scenario, event, label, target, trial_idx = epoch_info[i]
        epochs = trial_ds.windows

        epochs.events[:, -1] = target
        epochs.event_id = label_id
        epochs.metadata["target"] = target
        trial_ds.y = epochs.metadata.loc[:, "target"].to_list()

        # force update trial's description
        setattr(
            trial_ds,
            "_description",
            pd.Series(
                {
                    "scenario": scenario,
                    "event": event,
                    "label": label,
                    "trial": trial_idx,
                    "split": "train",
                }
            ),
        )

        if preload:
            epochs.load_data()

        # resample data after cropping
        if resample is not None:
            epochs.resample(resample, verbose=False)

    return ds


def load_data(
    data_dir: str,
    subjects: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    channels: Optional[List[str]] = None,
    label_mapping: Optional[Dict[str, str]] = None,
    minimal_trial_duration: float = 1.0,
    window_duration: float = 1.0,
    window_stride_duration: float = 0.25,
    start_offset: float = 0,
    stop_offset: float = 0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    moving_standardize: bool = True,
    resample: Optional[float] = None,
    return_raw: bool = False,
    return_preprocessed: bool = False,
    preload: bool = False,
) -> BaseConcatDataset:
    """
    load windowed variable-length trials from Official VIN dataset for cross subjects

    **Note**
        all "Resting" trials of each subject's runs are considered to be similar

    // all in seconds
                                                   trial onset +
                           trial onset          trial duration
       ----|--------------------|------------------------|-----------------------|----
           trial onset +                                             trial onset +
    trial start_offset                                            trial duration +
                                                               trial stop_offset

    Return
    ----------------------------
    ds      concat dataset
        concatenated dataset whose subdataset belongs to each subject's trial (raw's epochs)

        ds.description      pd dataframe
            subject
            scenario
            event
            trial_idx   index of trial on run (sample)
            split       split tag for braindecode's dataset splitting, in ["train", "valid", "test"]
    """

    subjects = set(subjects or list())

    # load raw datasets
    list_of_ds = list()
    for subject_dir in tqdm(sorted(glob.glob(f"{data_dir}/*")), desc="subject"):
        if not os.path.isdir(subject_dir) or not os.path.exists(
            f"{subject_dir}/info.json"
        ):
            continue

        subject_id = subject_dir.split("/")[-1]
        # load all "train" subjects if subjects is empty
        if len(subjects) > 0 and subject_id not in subjects:
            continue

        for sample_dir in sorted(glob.glob(f"{subject_dir}/sample*")):
            sample_ds = load_sample_data(
                sample_dir, scenarios, extra_description={"subject": subject_id}
            )

            if sample_ds is not None:
                list_of_ds.append(sample_ds)

    ds = BaseConcatDataset(list_of_ds)

    sfreq = ds.datasets[0].raw.info["sfreq"]
    # assert all(
    #     sample_ds.raw.info["sfreq"] == sfreq for sample_ds in ds.datasets
    # ), f"{data_dir}'s samples have different sampling frequencies"
    for sample_ds in ds.datasets:
        sample_sfreq = sample_ds.raw.info["sfreq"]
        assert sample_sfreq == sfreq, (
            f"{sample_ds.description['subject']}'s {sample_ds.description['scenario']} sample "
            f"doesn't have matched sampling frequency ({sample_sfreq}Hz, not {sfreq}Hz)"
        )

    if return_raw:
        return ds

    preprocess_data(ds, channels, fmin, fmax, moving_standardize)

    if return_preprocessed:
        return ds

    # create windows dataset
    mapping = {event: _id for _id, event in enumerate(events or DEFAULT_EVENTS)}
    label_mapping = label_mapping or dict()
    label_id = (
        {label_mapping.get("Resting", "Resting"): 0} if "Resting" in mapping else dict()
    )
    for scenario in sorted(ds.description["scenario"].unique()):
        for event in sorted(mapping.keys()):
            if event == "Resting":
                continue
            if scenario in INTENTION_SCENARIOS and event == "Thinking and Acting":
                continue

            label = f"{scenario}_{event}"
            label = label_mapping.get(label, label)
            if label in label_id:
                continue

            label_id[label] = len(label_id)

    list_of_epochs = list()
    epoch_info = list()
    ds_info = ds.description
    # minimal_trial_size = int(sfreq * minimal_trial_duration)
    for subject in tqdm(ds_info["subject"].unique(), desc="subject"):
        for i in ds_info[ds_info["subject"] == subject].index:
            sample_ds = ds.datasets[i]
            sample_annot = pd.DataFrame(sample_ds.raw.annotations)
            sample_events, sample_event_id = mne.events_from_annotations(
                sample_ds.raw, verbose=False
            )
            assert len(sample_annot) == len(sample_events)  # load all sample's events

            trial_count = dict()  # trial's index for each default label

            # due to variable-length trials, cannot load all epochs at the same time
            for i, r in sample_annot.iterrows():
                if r["description"] not in mapping:
                    continue

                if r["duration"] + stop_offset - start_offset < minimal_trial_duration:
                    continue

                # single epoch (1 event only)
                # tmin, tmax are relative to trial's onset (seconds)
                epochs = mne.Epochs(
                    sample_ds.raw,
                    sample_events[i : i + 1],
                    event_id=sample_event_id,
                    tmin=start_offset,
                    tmax=r["duration"] + stop_offset,
                    baseline=None,
                    preload=False,
                    proj=False,
                    on_missing="ignore",
                    event_repeated="drop",
                    verbose=False,
                )

                # if len(epochs.times) < minimal_trial_size:  # not enough data
                #     continue

                default_label = (
                    f"{sample_ds.description['scenario']}_{r['description']}"
                    if r["description"] != "Resting"
                    else "Resting"
                )
                label = label_mapping.get(default_label, default_label)

                if default_label not in trial_count:
                    trial_count[default_label] = 0

                list_of_epochs.append(epochs)
                epoch_info.append(
                    (
                        subject,
                        sample_ds.description["scenario"],
                        r["description"],
                        label,
                        label_id[label],
                        trial_count[default_label],
                    )
                )

                trial_count[default_label] += 1

    # split each epoch to overlapping sub-epochs (windows)
    ds = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=int(sfreq * window_duration),
        window_stride_samples=int(sfreq * window_stride_duration),
        drop_last_window=True,
    )

    # update trial's data and events
    if resample is not None:
        preload = True

    for i, trial_ds in tqdm(
        enumerate(ds.datasets), total=len(ds.datasets), desc="trial"
    ):
        subject, scenario, event, label, target, trial_idx = epoch_info[i]
        epochs = trial_ds.windows

        epochs.events[:, -1] = target
        epochs.event_id = label_id
        epochs.metadata["target"] = target
        trial_ds.y = epochs.metadata.loc[:, "target"].to_list()

        # force update trial's description
        setattr(
            trial_ds,
            "_description",
            pd.Series(
                {
                    "subject": subject,
                    "scenario": scenario,
                    "event": event,
                    "label": label,
                    "trial": trial_idx,
                    "split": "train",
                }
            ),
        )

        if preload:
            epochs.load_data()

        # resample data after cropping
        if resample is not None:
            epochs.resample(resample, verbose=False)

    return ds


def compute_transform_mat(
    X: np.ndarray, use_log: bool = False, inv: bool = True
) -> np.ndarray:
    """
    compute transform matrix (inv) sqrt of mean of covariances over trials

    Parameters
    ---------------------
    X   np.ndarray
        trials' data [bsz, channels, times]
    use_log     bool
        use log-euclidean metric or euclidean metric (default)
    inv     bool
        compute inverse of matrix (R^-1)

    Return
    ---------------------
    R   np.ndarray
        transform matrix [1, channels, channels]
    """
    assert X.ndim == 3, f"invalid input's ndim {X.ndim} isn't equal 3"

    Xm = X - X.mean(axis=2, keepdims=True)
    C = Xm @ Xm.transpose((0, 2, 1)) / (Xm.shape[2] - 1)  # [bsz, channels, channels]
    if use_log:  # log-euclid metric
        Cm = linalg.expm(np.mean(np.stack([linalg.logm(Ci) for Ci in C]), axis=0))
    else:  # euclid metric
        Cm = np.mean(C, axis=0)

    R = linalg.sqrtm(Cm)
    if inv:
        R = linalg.inv(R)
        if np.iscomplexobj(R):
            R = np.real(R).astype(np.float32)

    return R[np.newaxis]


def euclidean_alignment(
    ds: BaseConcatDataset,
    target_subject: Optional[str] = None,
    labeled_trials: Optional[Dict[str, List[int]]] = None,
    resting_label: Optional[str] = None,
):
    """
    apply euclidean alignment inplace for each subject of ds
    """
    ds_info = ds.description
    if target_subject is not None:
        assert (
            target_subject in ds_info["subject"].unique()
        ), f"target subject {target_subject} isn't in dataset"

    for subject in tqdm(ds_info["subject"].unique(), desc="subject"):
        if subject == target_subject and labeled_trials is not None:
            if resting_label is not None:
                labeled_trials = {resting_label: labeled_trials[resting_label]}

            subject_df = ds_info[ds_info["subject"] == target_subject]

            labeled_df = list()
            for label, trials in labeled_trials.items():
                labeled_df.append(
                    subject_df[
                        (subject_df["label"] == label)
                        & (subject_df["trial"].isin(trials))
                    ]
                )

            labeled_df = pd.concat(labeled_df)

            X = list()
            for i in labeled_df.index:
                X.append(ds.datasets[i].windows.get_data())

            X = np.concatenate(X, axis=0)  # [bsz, channels, times]
            R = compute_transform_mat(X)

            for i in subject_df.index:
                ds.datasets[i].windows.apply_function(
                    lambda _X: R @ _X, channel_wise=False
                )  # apply all channels at once

        else:
            subject_df = ds_info[ds_info["subject"] == subject]

            X = list()
            if resting_label is not None:
                for i in subject_df[subject_df["label"] == resting_label].index:
                    X.append(ds.datasets[i].windows.get_data())

            else:
                for i in subject_df.index:
                    X.append(ds.datasets[i].windows.get_data())

            X = np.concatenate(X, axis=0)  # [bsz, channels, times]
            R = compute_transform_mat(X)

            for i in subject_df.index:
                ds.datasets[i].windows.apply_function(
                    lambda _X: R @ _X, channel_wise=False
                )  # apply all channels at once


def label_alignment(
    ds: BaseConcatDataset,
    target_subject: str,
    labeled_trials: Optional[Dict[str, List[int]]] = None,
):
    """
    apply label alignment inplace for each subject of ds to target subject,
    only align "train" epochs
    """
    ds_info = ds.description
    assert (
        target_subject in ds_info["subject"].unique()
    ), f"target subject {target_subject} isn't in dataset"

    # compute target subject's transform matrices
    if labeled_trials is not None:
        target_ds_info = ds_info[
            (ds_info["subject"] == target_subject) & (ds_info["split"] == "train")
        ]

        tgt_df = list()
        for label, trials in labeled_trials.items():
            tgt_df.append(
                target_ds_info[
                    (target_ds_info["label"] == label)
                    & (target_ds_info["trial"].isin(trials))
                ]
            )

        tgt_idx = pd.concat(tgt_df).index
    else:
        tgt_idx = ds_info[
            (ds_info["subject"] == target_subject) & (ds_info["split"] == "train")
        ].index

    Xtgt = list()
    Ytgt = list()
    for i in tgt_idx:
        Xtgt.append(ds.datasets[i].windows.get_data())
        Ytgt.append(np.array(ds.datasets[i].y, dtype=int))

    Xtgt = np.concatenate(Xtgt, axis=0)
    Ytgt = np.concatenate(Ytgt, axis=0)

    tgtRs = dict()
    for y in np.unique(Ytgt):
        y_idx = np.where(Ytgt == y)[0]
        tgtRs[y] = compute_transform_mat(Xtgt[y_idx], inv=False)

    # apply for each subject in dataset
    for subject in tqdm(ds_info["subject"].unique(), desc="subject"):
        if subject == target_subject:
            continue

        src_idx = ds_info[
            (ds_info["subject"] == subject) & (ds_info["split"] == "train")
        ].index
        Xsrc = list()
        Ysrc = list()
        for i in src_idx:
            Xsrc.append(ds.datasets[i].windows.get_data())
            Ysrc.append(np.array(ds.datasets[i].y, dtype=int))

        Xsrc = np.concatenate(Xsrc, axis=0)
        Ysrc = np.concatenate(Ysrc, axis=0)

        srcRs = dict()
        for y in np.unique(Ysrc):
            y_idx = np.where(Ysrc == y)[0]
            srcRs[y] = tgtRs[y] @ compute_transform_mat(Xsrc[y_idx])

        for i in src_idx:
            y = ds.datasets[i].y[0]  # epochs have same label
            ds.datasets[i].windows.apply_function(
                lambda _X: srcRs[y] @ _X, channel_wise=False
            )  # appply all channels at once


if __name__ == "__main__":
    # for subject_dir in tqdm(
    #     sorted(glob.glob("../data/vin/Official/*")), desc="subject"
    # ):
    #     try:
    #         _ = load_subject_data(subject_dir, events=["Thinking"])

    #     except Exception as e:
    #         print(f"subject {subject_dir.split('/')[-1]} error!")

    ds = load_data("../data/vin/Official", events=["Thinking"], preload=True)
