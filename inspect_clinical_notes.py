import wfdb
import os

RECORDS = ['117', '214', '231', '111', '232']
DATA_PATH = r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

for rec in RECORDS:
    header = wfdb.rdheader(os.path.join(DATA_PATH, rec))
    print("=" * 60)
    print(f"CLINICAL RECORD NOTES FOR PATIENT {rec}:")
    print(f"  Signal shape: {header.sig_name}, Sampling freq: {header.fs} Hz")
    print(f"  Comments / Patient Diagnosis:")
    for comment in header.comments:
        print(f"    - {comment}")
    print("=" * 60 + "\n")
