import os
import pickle as pkl
import numpy as np
import pandas as pd
from moabb.paradigms import MotorImagery
from moabb.datasets import Cho2017, PhysionetMI, BNCI2014001
from braindecode.datasets import BaseConcatDataset, create_from_X_y

# ---------------------------------------------
# task 1: sleep stage
# ---------------------------------------------


# ---------------------------------------------
# task 2: motor imagery
# ---------------------------------------------


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
    os.makedirs("./data/beetl", exist_ok=True)

    # fmt: off
    hparams = {
        "resample": 128.0,
        "fmin": 4.0,
        "fmax": 38.0,
        # "ch_names": [   # exclude vin's channels ({FT9, FT10, PO9, PO10})
        #     "Fp1", "Fp2",
        #     "F7", "F3", "Fz", "F4", "F8",
        #     "FC5", "FC1", "FC2", "FC6",
        #     "T7", "C3", "Cz", "C4", "T8",
        #     "CP5", "CP1", "CP2", "CP6",
        #     "P7", "P3", "Pz", "P4", "P8",
        #     "O1", "Oz", "O2",
        # ],
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

    phy_hparams = {
        "ch_names": [
            'Fp1', 'Fpz', 'Fp2', 
            'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2',
            'Iz',
        ],
    }
    hparams.update(phy_hparams)

    # print("generate phy ds...")
    INVALID_SUBJECTS = {88, 92, 100, 104}  # missing/incorrect data
    # for s, e in [
    #     (1, 40), 
    #     (41, 80), 
    #     (81, 109),
    # ]:
    #     subjects = sorted(set(range(s, e + 1)) - INVALID_SUBJECTS)
    #     with open(f"./data/beetl/PHY_{s:03d}_{e:03d}.pkl", mode="wb") as f:
    #         pkl.dump(
    #             {"data": load_mi_data("phy", subjects=subjects, **hparams)},
    #             f,
    #         )

    # exit(0)

    print("create braindecode datasets...")
    # label_map = {"rest": 0, "feet": 1, "hands": 2, "left_hand": 3, "right_hand": 4}
    label_map = {"rest": 0, "feet": 1, "right_hand": 2, "left_hand": 3}  # LRF0
    # label_map = {"rest": 0, "right_hand": 1, "left_hand": 2}  # LRO

    for s, e in [(1, 40), (41, 80), (81, 109)]:
        with open(f"./data/beetl/PHY_{s:03d}_{e:03d}.pkl", mode="rb") as f:
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
                sfreq=128,
                ch_names=hparams["ch_names"],
                window_size_samples=int(128 * 3),  # duration: 3s
                window_stride_samples=int(128 * 3),  # stride: 3s
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
                            "label_idx": r["label_idx"],
                            "trial": int(r["run"].split("_")[1]),
                            "split": "train",
                        }
                    ),
                )

            list_of_ds.append(subject_ds)

        ds = BaseConcatDataset(list_of_ds)

        with open(f"./data/beetl/PHY_LRF0_{s:03d}_{e:03d}_4-38Hz_BRD.pkl", mode="wb") as f:
            pkl.dump(ds, f)
