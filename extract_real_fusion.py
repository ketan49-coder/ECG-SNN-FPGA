import wfdb
import numpy as np
import scipy.signal
import os

incart_records = ['I18', 'I04', 'I42', 'I05', 'I63', 'I75', 'I65', 'I46', 'I57', 'I64', 'I12', 'I43']
edb_records = ['e0605', 'e0411', 'e0609', 'e0305', 'e0509', 'e0203', 'e0602']
svdb_records = ['800', '806', '811', '803', '875']

extracted_beats = []
metadata = []

print("Extracting real Fusion beats from INCART (257 Hz -> 360 Hz)...")
for r in incart_records:
    try:
        record = wfdb.rdrecord(r, pn_dir='incartdb', channels=[0]) # Lead II / primary lead
        ann = wfdb.rdann(r, 'atr', pn_dir='incartdb')
        sig = record.p_signal.flatten()
        
        # Resample from 257 Hz to 360 Hz
        sig_360 = scipy.signal.resample(sig, int(len(sig) * 360.0 / 257.0))
        
        # Robust Median-MAD Normalization
        med = np.median(sig_360)
        mad = np.median(np.abs(sig_360 - med)) * 1.4826 + 1e-4
        sig_norm = np.clip(((sig_360 - med) / mad + 4.0) / 14.0, 0.0, 1.0)
        
        for idx, sym in enumerate(ann.symbol):
            if sym == 'F':
                sample_pos = int(ann.sample[idx] * 360.0 / 257.0)
                if sample_pos - 64 >= 0 and sample_pos + 64 <= len(sig_norm):
                    beat = sig_norm[sample_pos - 64 : sample_pos + 64]
                    if len(beat) == 128:
                        extracted_beats.append(beat)
                        metadata.append(f"INCART_{r}_{idx}")
    except Exception as e:
        print(f"  Error on INCART {r}: {e}")

print(f"Extracted {len(extracted_beats)} F beats from INCART.")

print("\nExtracting real Fusion beats from EDB (250 Hz -> 360 Hz)...")
for r in edb_records:
    try:
        record = wfdb.rdrecord(r, pn_dir='edb', channels=[0])
        ann = wfdb.rdann(r, 'atr', pn_dir='edb')
        sig = record.p_signal.flatten()
        
        # Resample 250 -> 360
        sig_360 = scipy.signal.resample_poly(sig, 36, 25)
        
        med = np.median(sig_360)
        mad = np.median(np.abs(sig_360 - med)) * 1.4826 + 1e-4
        sig_norm = np.clip(((sig_360 - med) / mad + 4.0) / 14.0, 0.0, 1.0)
        
        for idx, sym in enumerate(ann.symbol):
            if sym == 'F':
                sample_pos = int(ann.sample[idx] * 36.0 / 25.0)
                if sample_pos - 64 >= 0 and sample_pos + 64 <= len(sig_norm):
                    beat = sig_norm[sample_pos - 64 : sample_pos + 64]
                    if len(beat) == 128:
                        extracted_beats.append(beat)
                        metadata.append(f"EDB_{r}_{idx}")
    except Exception as e:
        print(f"  Error on EDB {r}: {e}")

print(f"Total after EDB: {len(extracted_beats)} F beats.")

print("\nExtracting real Fusion beats from SVDB (128 Hz -> 360 Hz)...")
for r in svdb_records:
    try:
        record = wfdb.rdrecord(r, pn_dir='svdb', channels=[0])
        ann = wfdb.rdann(r, 'atr', pn_dir='svdb')
        sig = record.p_signal.flatten()
        
        # Resample 128 -> 360
        sig_360 = scipy.signal.resample_poly(sig, 45, 16)
        
        med = np.median(sig_360)
        mad = np.median(np.abs(sig_360 - med)) * 1.4826 + 1e-4
        sig_norm = np.clip(((sig_360 - med) / mad + 4.0) / 14.0, 0.0, 1.0)
        
        for idx, sym in enumerate(ann.symbol):
            if sym == 'F':
                sample_pos = int(ann.sample[idx] * 45.0 / 16.0)
                if sample_pos - 64 >= 0 and sample_pos + 64 <= len(sig_norm):
                    beat = sig_norm[sample_pos - 64 : sample_pos + 64]
                    if len(beat) == 128:
                        extracted_beats.append(beat)
                        metadata.append(f"SVDB_{r}_{idx}")
    except Exception as e:
        print(f"  Error on SVDB {r}: {e}")

all_f_beats = np.array(extracted_beats, dtype=np.float32)
np.savez_compressed('all_real_fusion_beats.npz', X_fusion=all_f_beats, metadata=np.array(metadata))

print(f"\n=======================================================")
print(f"SUCCESS: Extracted {len(all_f_beats)} REAL Fusion beats!")
print(f"Shape: {all_f_beats.shape} | Saved to 'all_real_fusion_beats.npz'")
print(f"=======================================================")
