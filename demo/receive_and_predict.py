"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

import argparse
import random
import time

import numpy as np
import pylsl


def receive_and_predict(
    inlet_stream, marker_outlet_stream, window_size, predict_interval
):
    # create a new inlet to read from the stream
    inlet = None
    for stream in pylsl.resolve_stream():
        if stream.name() == inlet_stream:
            inlet = pylsl.StreamInlet(stream)
            break

    assert inlet is not None, f"invalid inlet {inlet_stream}"

    # create a new marker outlet for prediction
    info = pylsl.StreamInfo(
        name=marker_outlet_stream,
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
    )
    marker_outlet = pylsl.StreamOutlet(info)

    # load model and labels
    labels = [f"label_{i}" for i in range(5)]

    n_channel = inlet.info().channel_count()
    n_proc_samples = int(window_size * inlet.info().nominal_srate())
    buffer = np.zeros((n_channel, 0), dtype=np.float32)

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        chunk, timestamps = inlet.pull_chunk()
        if timestamps:
            chunk = np.array(chunk, dtype=np.float32).T  # [n_channels,X]
            buffer = np.hstack((buffer, chunk))[
                :, -n_proc_samples:
            ]  # [n_channels,pX+X]
            if buffer.shape[1] == n_proc_samples:
                # make prediction
                if random.getrandbits(1):
                    marker_outlet.push_sample([random.choice(labels)])

            time.sleep(predict_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inlet_stream",
        type=str,
        default="EEGInputStream",
        help="name of eeg inlet stream",
    )
    parser.add_argument(
        "--marker_outlet_stream",
        type=str,
        default="PredictionMarkerStream",
        help="name of prediction marker stream",
    )
    parser.add_argument(
        "--window_size", type=int, default=2, help="window size for prediction"
    )
    parser.add_argument(
        "--predict_interval",
        type=int,
        default=250,
        help="duration between 2 consecutive predictions (ms)",
    )
    arg = parser.parse_args()

    receive_and_predict(
        arg.inlet_stream,
        arg.marker_outlet_stream,
        arg.window_size,
        arg.predict_interval / 1000.0,
    )
