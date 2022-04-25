import argparse
import mne
import pylsl
import time


# fmt: off
default_ch_names = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FT9", "FC5", "FC1", "FC2", "FC6", "FT10",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]
# fmt: on


def send_data(edf_path, outlet_stream, ch_names=None):
    # read EEG data from edf file
    raw: mne.io.Raw = mne.io.read_raw_edf(edf_path, preload=False)

    ch_names = ch_names or default_ch_names
    n_channels = len(ch_names)

    srate = raw.info["sfreq"]

    # create pylsl stream
    info = pylsl.StreamInfo(
        name=outlet_stream,
        type="EEG",
        channel_count=n_channels,
        nominal_srate=srate,
        channel_format="float32",
    )
    outlet = pylsl.StreamOutlet(info, chunk_size=32, max_buffered=360)  # params?

    # send data
    print(f"sending data from {edf_path}...")

    start_time = pylsl.local_clock()
    sent_samples = 0
    chunk_start = 0
    while True:
        elapsed_time = pylsl.local_clock() - start_time
        required_samples = (
            int(srate * elapsed_time) - sent_samples
        )  # num of samples need to send

        if required_samples > 0:
            try:
                # fetch chunk from raw
                chunk = raw.get_data(
                    picks=ch_names,
                    start=chunk_start,
                    stop=chunk_start + required_samples,
                    verbose=False,
                ).T  # [n_samples, n_channels]

            except Exception as err:
                print(f"error: {err} - restart sending data...")

                # restart fetching data
                chunk_start = 0
                chunk = raw.get_data(
                    picks=ch_names,
                    start=chunk_start,
                    stop=chunk_start + required_samples,
                    verbose=False,
                ).T  # [n_samples, n_channels]

            stamp = pylsl.local_clock()
            outlet.push_chunk(chunk.tolist(), stamp)

            sent_samples += required_samples
            chunk_start += required_samples

        time.sleep(0.1)  # need to send x*srate samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outlet_stream",
        type=str,
        default="EEGInputStream",
        help="name of eeg outlet stream",
    )
    parser.add_argument("--edf_path", required=True, help="edf file")
    arg = parser.parse_args()

    send_data(arg.edf_path, arg.outlet_stream)
