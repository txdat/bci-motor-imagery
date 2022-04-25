import logging
import numpy as np
import pandas as pd
import mne
from braindecode.preprocessing.preprocess import exponential_moving_standardize
from moabb.paradigms import MotorImagery
from moabb.datasets import Cho2017, PhysionetMI, BNCI2014001

logging.getLogger("mne").setLevel(logging.ERROR)

# ---------------------------------------------
# task 1: sleep stage
# ---------------------------------------------


# ---------------------------------------------
# task 2: motor imagery
# ---------------------------------------------


class StandardizedMotorImagery(MotorImagery):
    def process_raw(self, raw, dataset, return_epochs=False):
        # get events id
        event_id = self.used_events(dataset)

        # find the events, first check stim_channels then annotations
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            try:
                events, _ = mne.events_from_annotations(
                    raw, event_id=event_id, verbose=False
                )
            except ValueError:
                logging.warning("No matching annotations in {}".format(raw.filenames))
                return

        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                raw.info["ch_names"], include=self.channels, ordered=True
            )

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            raw_f = raw.copy().filter(
                fmin,
                fmax,
                method="fir",
                picks=picks,
                # iir_params={"order": 5, "ftype": "butter"},
                verbose=False,
            )

            # convert signal from V to uV (unit_factor is 1e6)
            raw_f = raw_f.apply_function(lambda x: x * dataset.unit_factor, picks=picks)

            # standardize data
            raw_f = raw_f.apply_function(
                lambda x: exponential_moving_standardize(
                    x, factor_new=1e-3, init_block_size=1000
                ),
                picks=picks,
                channel_wise=False,
            )

            # epoch data
            baseline = self.baseline
            if baseline is not None:
                baseline = (
                    self.baseline[0] + dataset.interval[0],
                    self.baseline[1] + dataset.interval[0],
                )
                bmin = baseline[0] if baseline[0] < tmin else tmin
                bmax = baseline[1] if baseline[1] > tmax else tmax
            else:
                bmin = tmin
                bmax = tmax

            epochs = mne.Epochs(
                raw_f,
                events,
                event_id=event_id,
                tmin=bmin,
                tmax=bmax,
                proj=False,
                baseline=baseline,
                preload=True,
                verbose=False,
                picks=picks,
                event_repeated="drop",
                on_missing="ignore",
            )

            if bmin < tmin or bmax > tmax:
                epochs.crop(tmin=tmin, tmax=tmax)

            # resample data to target frequency
            # https://mne.tools/stable/overview/faq.html#resampling-and-decimating
            if self.resample is not None:
                epochs = epochs.resample(self.resample)

            if return_epochs:  # TODO: recheck
                # rescale signal from uV to V
                epochs = epochs.apply_function(lambda x: x / dataset.unit_factor)
                X.append(epochs)
            else:
                X.append(epochs.get_data())

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        if return_epochs:
            X = mne.concatenate_epochs(X)
        elif len(self.filters) == 1:
            # if only one band, return a 3D array
            X = X[0]
        else:
            # otherwise return a 4D
            X = np.array(X).transpose((1, 2, 3, 0))

        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata


def load_mi_data(
    ds_name, subjects, fmin, fmax, resample=None, ch_names=None, return_epochs=False
):
    if ds_name == "cho":
        ds = Cho2017()

    elif ds_name == "phy":
        ds = PhysionetMI(imagined=True, executed=False)

    elif ds_name == "bnci":
        ds = BNCI2014001()

    else:
        raise ValueError(f"invalid dataset's name: {ds_name}")

    mi = MotorImagery(
        n_classes=len(ds.event_id),
        channels=ch_names,
        fmin=fmin,
        fmax=fmax,
        resample=resample,
    )

    return mi.get_data(ds, subjects, return_epochs=return_epochs)


if __name__ == "__main__":
    import pickle as pkl
    from braindecode.datasets import BaseConcatDataset, create_from_X_y

    # fmt: off
    hparams = {
        "resample": 128.0,
        "fmin": 8.0,
        "fmax": 30.0,
        "ch_names": [   # exclude vin's channels ({FT9, FT10, PO9, PO10})
            "Fp1", "Fp2",
            "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6",
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz", "O2",
        ],
    }
    # fmt: on

    # print("generate bnci ds...")
    # # fmt:off
    # bnci_hparams = {"ch_names": [
    #     'Fz',
    #     'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    #     'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    #     'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    #     'P1', 'Pz', 'P2',
    #     'POz',
    # ]}
    # # fmt:on
    # with open("./data/beetl/bnci.pkl", mode="wb") as f:
    #     pkl.dump(
    #         {
    #             "data": load_mi_data(
    #                 "bnci",
    #                 subjects=list(range(1, 10)),
    #                 **{**hparams, **bnci_hparams},
    #                 return_epochs=False,
    #             )
    #         },
    #         f,
    #     )

    # print("generate cho ds...")
    # INVALID_SUBJECTS = {32, 46, 49}
    # subjects = sorted(set(range(1, 53)) - INVALID_SUBJECTS)

    # with open("./data/beetl/cho.pkl", mode="wb") as f:
    #     pkl.dump({"data": load_mi_data("cho", subjects=subjects, **hparams)}, f)

    print("generate phy ds...")
    INVALID_SUBJECTS = {88, 92, 100, 104}  # missing/incorrect data
    # for s, e in [(1, 40), (41, 80), (81, 109)]:
    #     subjects = sorted(set(range(s, e + 1)) - INVALID_SUBJECTS)
    #     with open(f"./data/beetl/phy_vin_{s}_{e}.pkl", mode="wb") as f:
    #         pkl.dump(
    #             {"data": load_mi_data("phy", subjects=subjects, **hparams)},
    #             f,
    #         )

    print("create braindecode datasets...")
    # label_map = {"rest": 0, "feet": 1, "hands": 2, "left_hand": 3, "right_hand": 4}
    # label_map = {"rest": 0, "feet": 1, "right_hand": 2, "left_hand": 3}
    label_map = {"rest": 0, "right_hand": 1, "left_hand": 2}

    for s, e in [(1, 40), (41, 80), (81, 109)]:
        with open(f"./data/beetl/phy_vin_{s}_{e}.pkl", mode="rb") as f:
            X, labels, meta = pkl.load(f)["data"]
            meta["label"] = labels

        meta = meta[
            (~meta["subject"].isin(INVALID_SUBJECTS)) & (meta["label"].isin(label_map))
        ]
        X = X[meta.index]
        meta = meta.reset_index(drop=True)
        meta["label_idx"] = 0
        for i, r in meta.iterrows():
            meta.loc[i, "label_idx"] = label_map[r["label"]]

        list_of_ds = list()
        for subject in meta["subject"].unique():
            subject_meta = meta[meta["subject"] == subject]
            x = X[subject_meta.index]
            y = np.array(subject_meta["label_idx"], dtype=int)
            subject_meta = subject_meta.reset_index(drop=True)

            subject_ds = create_from_X_y(
                x,
                y,
                drop_last_window=True,
                sfreq=128.0,
                ch_names=hparams["ch_names"],
                window_size_samples=256,
                window_stride_samples=64,
            )
            for i, ds in enumerate(subject_ds.datasets):
                r = subject_meta.iloc[i]

                setattr(
                    ds,
                    "_description",
                    pd.Series(
                        {
                            "subject": f"PHY_{subject:03d}",
                            "scenario": r["label"],
                            "event": "Resting" if r["label"] == "rest" else "Thinking",
                            "label": r["label"],
                            "trial": int(r["run"].split("_")[1]),
                            "split": "train",
                        }
                    ),
                )

            list_of_ds.append(subject_ds)

        ds = BaseConcatDataset(list_of_ds)

        with open(f"./data/beetl/phy_vin_LR_{s}_{e}_braindcd.pkl", mode="wb") as f:
            pkl.dump(ds, f)
