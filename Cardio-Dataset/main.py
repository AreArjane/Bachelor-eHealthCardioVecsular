import wfdb
from pathlib import Path

def get_sampling_rate(header_file: Path):
    """Read sampling rate (Hz) from a .hea header file."""
    base = header_file.with_suffix('')  # remove .hea
    header = wfdb.rdheader(str(base))
    return header.fs

def scan_dataset(folder: str, dataset_name: str):
    """Scan all .hea files in a folder and print results neatly."""
    folder = Path(folder)
    print("\n" + "=" * 60)
    print(f"📂 Dataset: {dataset_name}")
    print("=" * 60)
    for hea_file in sorted(folder.glob("*.hea")):
        try:
            fs = get_sampling_rate(hea_file)
            print(f"🩺 Record: {hea_file.stem:<10} | Sampling Frequency: {fs:>6.1f} Hz")
        except Exception as e:
            print(f"⚠️  {hea_file.stem:<10} | Failed to read -> {e}")
    print("=" * 60 + "\n")

# Example usage:
scan_dataset("files/", "MIT-BIH Atrial Fibrillation Database")
scan_dataset("mit-bih-normal-sinus-rhythm-database-1.0.0/",  "MIT-BIH Normal Sinus Rhythm Database")

