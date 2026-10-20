import wfdb
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

def check_rec_incart(r):
    try:
        ann = wfdb.rdann(r, 'atr', pn_dir='incartdb')
        c = ann.symbol.count('F')
        return r, c
    except Exception:
        return r, 0

def check_rec_edb(r):
    try:
        ann = wfdb.rdann(r, 'atr', pn_dir='edb')
        c = ann.symbol.count('F')
        return r, c
    except Exception:
        return r, 0

def check_rec_svdb(r):
    try:
        ann = wfdb.rdann(r, 'atr', pn_dir='svdb')
        c = ann.symbol.count('F')
        return r, c
    except Exception:
        return r, 0

print("Scanning INCART, EDB, and SVDB in parallel...")

# 1. INCART
incart_recs = wfdb.get_record_list('incartdb')
incart_f = {}
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(check_rec_incart, r) for r in incart_recs]
    for fut in as_completed(futs):
        r, c = fut.result()
        if c > 0:
            incart_f[r] = c
print(f"\nINCART ({len(incart_recs)} records): {sum(incart_f.values())} F beats across {len(incart_f)} records: {incart_f}")

# 2. SVDB
svdb_recs = wfdb.get_record_list('svdb')
svdb_f = {}
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(check_rec_svdb, r) for r in svdb_recs]
    for fut in as_completed(futs):
        r, c = fut.result()
        if c > 0:
            svdb_f[r] = c
print(f"\nSVDB ({len(svdb_recs)} records): {sum(svdb_f.values())} F beats across {len(svdb_f)} records: {svdb_f}")

# 3. EDB
edb_recs = wfdb.get_record_list('edb')
edb_f = {}
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(check_rec_edb, r) for r in edb_recs]
    for fut in as_completed(futs):
        r, c = fut.result()
        if c > 0:
            edb_f[r] = c
print(f"\nEDB ({len(edb_recs)} records): {sum(edb_f.values())} F beats across {len(edb_f)} records: {edb_f}")

total_extra_f = sum(incart_f.values()) + sum(svdb_f.values()) + sum(edb_f.values())
print(f"\n==================================================")
print(f"TOTAL EXTRA FUSION BEATS FOUND: {total_extra_f}")
print(f"==================================================")
