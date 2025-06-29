import argparse
import os
import pandas as pd
import soundfile as sf


def fix_end_times(csv_path: str, output_path: str = None) -> str:
    """Update rows where ``end_time`` is zero using the audio clip duration.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing at least ``start_time``, ``end_time``
        and ``audio_clip_path`` columns.
    output_path : str, optional
        Where to write the updated CSV. Defaults to ``segments_fixed.csv`` in
        the same directory as ``csv_path``.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    df = pd.read_csv(csv_path)

    if output_path is None:
        directory, name = os.path.split(csv_path)
        output_path = os.path.join(directory, "segments_fixed.csv")

    for idx, row in df[df.get("end_time", 0) == 0].iterrows():
        audio_file = row.get("audio_clip_path")
        if not audio_file or not os.path.exists(audio_file):
            continue
        with sf.SoundFile(audio_file) as f:
            duration = len(f) / f.samplerate
        df.loc[idx, "end_time"] = row["start_time"] + duration

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix zero end_time values in a segments CSV")
    parser.add_argument("csv", help="Path to segments CSV")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    result = fix_end_times(args.csv, args.output)
    print(f"Wrote updated segments to {result}")
