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
#     0-------X-------|Thinking|-------X--------|Resting|---------X----------|Thinking and Acting|------
#     --------X-------|Thinking|-------X--------|Resting|---------X----------|Thinking and Acting|------
#     --------X-------|Thinking|-------X--------|Resting|---------X----------|Thinking and Acting|------
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
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> None:
    """
    preprocess raw data inplace
    """

    # init preprocessor
    preprocessors = [
        # Preprocessor("pick_types", eeg=True, stim=False),  # Keep EEG sensors
        Preprocessor(
            "pick_channels",
            ch_names=channels or DEFAULT_CHANNELS,
            ordered=True,
        ),
        Preprocessor(
            "filter",
            l_freq=fmin,
            h_freq=fmax,
            n_jobs=-1,
            verbose=False,
        ),  # Bandpass filter using fir
        Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),  # Exponential moving standardization
    ]

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

    raw = mne.io.read_raw_edf(f"{sample_dir}/EEG.edf", preload=True, verbose=False)
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
    window_duration: float = 2.0,
    window_stride_duration: float = 0.25,
    start_offset: float = 0,
    stop_offset: float = 0,
    fmin: float = 8.0,
    fmax: float = 30.0,
    resample: Optional[float] = None,
    return_raw: bool = False,
) -> BaseConcatDataset:
    """
       load windowed variable-length trials from Official VIN dataset for target subject

       // all in seconds
                                                   trial onset +
                           trial onset          trial duration
       ----|--------------------|------------------------|-----------------------|----
           trial onset +                                             trial onset +
    trial start_offset                                            trial duration +
                                                               trial stop_offset
    """

    # load raw datasets
    # each dataset belongs to sample
    list_of_ds = list()
    for sample_dir in tqdm(sorted(glob.glob(f"{subject_dir}/sample*")), desc="sample"):
        sample_ds = load_sample_data(sample_dir, scenarios)
        if sample_ds is not None:
            list_of_ds.append(sample_ds)

    ds = BaseConcatDataset(list_of_ds)

    sfreq = ds.datasets[0].raw.info["sfreq"]
    assert all(
        sample_ds.raw.info["sfreq"] == sfreq for sample_ds in ds.datasets
    ), f"{subject_dir}'s samples have different sampling frequencies"

    if return_raw:
        return ds

    preprocess_data(ds, channels, fmin, fmax)

    # create windows dataset
    window_size = int(sfreq * window_duration)
    window_stride = int(sfreq * window_stride_duration)

    mapping = {event: _id for _id, event in enumerate(events or DEFAULT_EVENTS)}
    label_id = {"Resting": 0} if "Resting" in mapping else dict()
    for scenario in sorted(ds.description["scenario"].unique()):
        for event in sorted(mapping.keys()):
            if event == "Resting":
                continue
            if scenario in INTENTION_SCENARIOS and event == "Thinking and Acting":
                continue

            label_id[f"{scenario}_{event}"] = len(label_id)

    # split trials from samples before cropping to windows
    list_of_epochs = list()
    epoch_info = list()
    trial_count = dict()  # trial's index for each label
    for sample_ds in tqdm(ds.datasets, desc="sample"):
        sample_annot = pd.DataFrame(sample_ds.raw.annotations)
        sample_events, sample_event_id = mne.events_from_annotations(
            sample_ds.raw, verbose=False
        )
        assert len(sample_annot) == len(sample_events)  # load all sample's events

        # due to variable-length trials, cannot load all epochs at the same time
        for i, r in sample_annot.iterrows():
            if r["description"] not in mapping:
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
                preload=True,
                proj=False,
                on_missing="ignore",
                event_repeated="drop",
                verbose=False,
            )

            if len(epochs.times) < window_size:  # not enough data
                continue

            label = (
                f"{sample_ds.description['scenario']}_{r['description']}"
                if r["description"] != "Resting"
                else "Resting"
            )
            if label not in trial_count:
                trial_count[label] = 0

            list_of_epochs.append(epochs)
            epoch_info.append(
                (
                    sample_ds.description["scenario"],
                    r["description"],
                    label_id[label],
                    trial_count[label],
                )
            )

            trial_count[label] += 1

    # split each epoch to overlapping sub-epochs (windows)
    ds = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=window_size,
        window_stride_samples=window_stride,
        drop_last_window=True,
    )

    # update trial's data and events
    for i, trial_ds in tqdm(
        enumerate(ds.datasets), total=len(ds.datasets), desc="trial"
    ):
        scenario, event, target, trial_idx = epoch_info[i]
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
                    "trial": trial_idx,
                    "split": "train",
                }
            ),
        )

        # resample data after cropping
        if resample is not None:
            epochs.load_data()
            epochs.resample(resample, verbose=False)

    return ds


def load_data(
    data_dir: str,
    subjects: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    channels: Optional[List[str]] = None,
    window_duration: float = 2.0,
    window_stride_duration: float = 0.25,
    start_offset: float = 0,
    stop_offset: float = 0,
    fmin: float = 8.0,
    fmax: float = 30.0,
    resample: Optional[float] = None,
    return_raw: bool = False,
) -> BaseConcatDataset:
    """
       load windowed variable-length trials from Official VIN dataset for cross subjects

       // all in seconds
                                                   trial onset +
                           trial onset          trial duration
       ----|--------------------|------------------------|-----------------------|----
           trial onset +                                             trial onset +
    trial start_offset                                            trial duration +
                                                               trial stop_offset
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
    assert all(
        sample_ds.raw.info["sfreq"] == sfreq for sample_ds in ds.datasets
    ), f"{data_dir}'s samples have different sampling frequencies"

    if return_raw:
        return ds

    preprocess_data(ds, channels, fmin, fmax)

    # create windows dataset
    window_size = int(sfreq * window_duration)
    window_stride = int(sfreq * window_stride_duration)

    mapping = {event: _id for _id, event in enumerate(events or DEFAULT_EVENTS)}
    label_id = {"Resting": 0} if "Resting" in mapping else dict()
    for scenario in sorted(ds.description["scenario"].unique()):
        for event in sorted(mapping.keys()):
            if event == "Resting":
                continue
            if scenario in INTENTION_SCENARIOS and event == "Thinking and Acting":
                continue

            label_id[f"{scenario}_{event}"] = len(label_id)

    list_of_epochs = list()
    epoch_info = list()
    ds_info = ds.description
    for subject in tqdm(ds_info["subject"].unique(), desc="subject"):
        trial_count = dict()  # trial's index for each label
        for i in ds_info[ds_info["subject"] == subject].index:
            sample_ds = ds.datasets[i]
            sample_annot = pd.DataFrame(sample_ds.raw.annotations)
            sample_events, sample_event_id = mne.events_from_annotations(
                sample_ds.raw, verbose=False
            )
            assert len(sample_annot) == len(sample_events)  # load all sample's events

            # due to variable-length trials, cannot load all epochs at the same time
            for i, r in sample_annot.iterrows():
                if r["description"] not in mapping:
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
                    preload=True,
                    proj=False,
                    on_missing="ignore",
                    event_repeated="drop",
                    verbose=False,
                )

                if len(epochs.times) < window_size:  # not enough data
                    continue

                label = (
                    f"{sample_ds.description['scenario']}_{r['description']}"
                    if r["description"] != "Resting"
                    else "Resting"
                )
                if label not in trial_count:
                    trial_count[label] = 0

                list_of_epochs.append(epochs)
                epoch_info.append(
                    (
                        subject,
                        sample_ds.description["scenario"],
                        r["description"],
                        label_id[label],
                        trial_count[label],
                    )
                )

                trial_count[label] += 1

    # split each epoch to overlapping sub-epochs (windows)
    ds = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=window_size,
        window_stride_samples=window_stride,
        drop_last_window=True,
    )

    # update trial's data and events
    for i, trial_ds in tqdm(
        enumerate(ds.datasets), total=len(ds.datasets), desc="trial"
    ):
        subject, scenario, event, target, trial_idx = epoch_info[i]
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
                    "trial": trial_idx,
                    "split": "train",
                }
            ),
        )

        # resample data after cropping
        if resample is not None:
            epochs.load_data()
            epochs.resample(resample, verbose=False)

    return ds


if __name__ == "__main__":
    # fmt: off
    channels = [
        "Fp1", "F7", "F3", "FC5", "FC1", "T7", "C3", "CP5", "CP1", "P7", "P3", "O1",  # left
        "Fz", "Cz", "Pz", "Oz",  # center
        "Fp2", "F8", "F4", "FC6", "FC2", "T8", "C4", "CP6", "CP2", "P8", "P4", "O2",  # right
    ]
    # fmt: on

    # for subject_dir in tqdm(
    #     sorted(glob.glob("../data/vin/Official/*")), desc="subject"
    # ):
    #     try:
    #         _ = load_subject_data(subject_dir, events=["Thinking"], channels=channels)
    #
    #     except Exception as e:
    #         print(f"subject {subject_dir.split('/')[-1]} error!")

    _ = load_data("../data/vin/Official", events=["Thinking"], channels=channels)
