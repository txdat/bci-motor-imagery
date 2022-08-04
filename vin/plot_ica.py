import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import mne
import json
import glob
import os
from tqdm import tqdm
import time
from datetime import timedelta


for subject in tqdm(sorted(glob.glob("../data/DataVIN/Official/*")), desc="subject"):
    for sample in sorted(glob.glob(f"{subject}/sample*")):
        if not os.path.isdir(sample):
            continue

        try:
            # ---------------raw and annotation fixing---------------
            with open(f"{sample}/eeg.json", mode="r", encoding="utf-8") as f:
                scenario = json.load(f).get("Scenario", "").strip().lower()

                # fix scenario's name
                if scenario == "nang tay trai":
                    scenario = "nâng tay trái"

                intention_scenario = scenario in {
                    "tôi muốn uống nước",
                    "tôi muốn vệ sinh",
                }

            raw = mne.io.read_raw_edf(f"{sample}/EEG.edf", preload=True, verbose=False)

            # fix wrong annotation in some subjects
            # TODO: double check?
            if intention_scenario:
                raw.annotations.description[
                    np.where(raw.annotations.description == "Thinking and Acting")[0]
                ] = "Thinking"

            if sample.endswith("K312/sample1"):
                # wrong label "Thinking" (2)
                raw.annotations.description[2] = "Thinking and Acting"

            elif sample.endswith("K360/sample5"):
                # wrong label "Thinking and Acting" (8)
                raw.annotations.description[8] = "Thinking"

            elif sample.endswith("K369/sample6"):
                # wrong label "Thinking and Acting" (8)
                raw.annotations.description[8] = "Thinking"

            df = pd.DataFrame(raw.annotations)
            df["onset (1)"] = df["onset"].apply(lambda x: timedelta(seconds=x))
            df["offset (1)"] = df.apply(
                lambda r: timedelta(seconds=r["onset"] + r["duration"]), axis=1
            )
            print(df)

            # ---------------ica validation--------------------------
            raw = raw.filter(l_freq=8.0, h_freq=30.0, verbose=False)

            ica = mne.preprocessing.ICA(
                n_components=32, max_iter=5000, random_state=42, verbose=False
            )
            ica.fit(raw)

            fig = ica.plot_sources(raw, picks=list(range(32)))
            # plt.savefig(f"../data/DataVIN/Official_ICA/{subject[subject.rfind('/') + 1:]}_{sample[sample.rfind('/') + 1:]}.png")
            # plt.close(fig)

        except Exception as e:
            print(f"error {sample}:\t{e}")

        break
    break
