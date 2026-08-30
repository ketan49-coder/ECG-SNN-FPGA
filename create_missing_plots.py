import json
import numpy as np
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0")

# aami_distribution.json
with open('../aami_distribution.json', 'r') as f:
    aami_data = json.load(f)

overall = aami_data['overall']

# aami_pie.png
main_classes = ['N', 'S', 'V', 'F']
main_counts = [overall[c] for c in main_classes]
main_labels = ['Normal', 'SVEB', 'VEB', 'Fusion']
colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(main_counts, labels=main_labels, colors=colors[:4],
                                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_color('white')
ax.set_title('AAMI Beat Classes (Excluding Unknown/Non-beat)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../aami_pie.png', dpi=150, facecolor='#fcfcfb')
print("Saved aami_pie.png")

# beat_morphology.png
def extract_beats(record_name, lead_idx=0, window_before=90, window_after=150):
    record = wfdb.rdrecord(record_name)
    ann = wfdb.rdann(record_name, 'atr')
    signal = record.p_signal[:, lead_idx]
    beats_by_class = {}
    samples = ann.sample
    symbols = ann.symbol
    for s, sym in zip(samples, symbols):
        start = s - window_before
        end = s + window_after
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            if sym not in beats_by_class:
                beats_by_class[sym] = []
            beats_by_class[sym].append(beat)
    return beats_by_class, record.sig_name[lead_idx], record.fs

beats_100, _, _ = extract_beats('100', 0)
beats_105, _, _ = extract_beats('105', 0)
beats_200, _, _ = extract_beats('200', 0)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
t = np.arange(-90, 150) / 360 * 1000

records_data = [('100', beats_100), ('105', beats_105), ('200', beats_200)]

for row_idx, (rec_name, beats_dict) in enumerate(records_data):
    for col_idx, (cls, color, title) in enumerate([('N', '#2a78d6', 'Normal'), ('V', '#1baf7a', 'VEB'), ('A', '#eb6834', 'SVEB')]):
        ax = axes[row_idx, col_idx]
        if cls in beats_dict and len(beats_dict[cls]) > 0:
            avg_beat = np.mean(beats_dict[cls], axis=0)
            std_beat = np.std(beats_dict[cls], axis=0)
            ax.plot(t, avg_beat, color=color, linewidth=1.5, label=f'{title} (n={len(beats_dict[cls])})')
            ax.fill_between(t, avg_beat - std_beat, avg_beat + std_beat, color=color, alpha=0.2)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title('Record {} - {}'.format(rec_name, title), fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (ms)', fontsize=8)
            ax.set_ylabel('mV', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)
        else:
            ax.set_title('Record {} - {} (none)'.format(rec_name, title), fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)

plt.suptitle('Average Beat Morphology by Class and Record', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../beat_morphology.png', dpi=150, facecolor='#fcfcfb')
print("Saved beat_morphology.png")

# signal_heatmap.png
df_stats = pd.read_csv('../signal_stats.csv')
pivot_mean = df_stats.pivot(index='record', columns='lead', values='mean')
pivot_std = df_stats.pivot(index='record', columns='lead', values='std')
pivot_range = df_stats.pivot(index='record', columns='lead', values='range')

fig, axes = plt.subplots(1, 3, figsize=(18, 12))
im1 = axes[0].imshow(pivot_mean.values, aspect='auto', cmap='RdBu_r', interpolation='nearest')
axes[0].set_xticks(range(len(pivot_mean.columns)))
axes[0].set_xticklabels(pivot_mean.columns, fontsize=10)
axes[0].set_yticks(range(len(pivot_mean.index)))
axes[0].set_yticklabels(pivot_mean.index, fontsize=8)
axes[0].set_title('Mean Signal (mV)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(pivot_std.values, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1].set_xticks(range(len(pivot_std.columns)))
axes[1].set_xticklabels(pivot_std.columns, fontsize=10)
axes[1].set_yticks(range(len(pivot_std.index)))
axes[1].set_yticklabels(pivot_std.index, fontsize=8)
axes[1].set_title('Std Dev (mV)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(pivot_range.values, aspect='auto', cmap='plasma', interpolation='nearest')
axes[2].set_xticks(range(len(pivot_range.columns)))
axes[2].set_xticklabels(pivot_range.columns, fontsize=10)
axes[2].set_yticks(range(len(pivot_range.index)))
axes[2].set_yticklabels(pivot_range.index, fontsize=8)
axes[2].set_title('Signal Range (mV)', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=axes[2])

plt.suptitle('Signal Statistics Across Records and Leads', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../signal_heatmap.png', dpi=150, facecolor='#fcfcfb')
print("Saved signal_heatmap.png")

# rr_analysis.png
df_rr = pd.read_csv('../rr_intervals.csv')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].bar(df_rr['record'], df_rr['rr_mean'], color='#2a78d6', edgecolor='white', linewidth=0.5)
axes[0, 0].set_ylabel('Mean RR (ms)', fontsize=10)
axes[0, 0].set_title('Mean RR Interval per Record', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0, 0].grid(axis='y', alpha=0.3)

axes[0, 1].bar(df_rr['record'], df_rr['hr_mean'], color='#1baf7a', edgecolor='white', linewidth=0.5)
axes[0, 1].set_ylabel('Mean HR (bpm)', fontsize=10)
axes[0, 1].set_title('Mean Heart Rate per Record', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[0, 1].grid(axis='y', alpha=0.3)

cv = df_rr['rr_std'] / df_rr['rr_mean'] * 100
axes[1, 0].bar(df_rr['record'], cv, color='#eb6834', edgecolor='white', linewidth=0.5)
axes[1, 0].set_ylabel('CV of RR (%)', fontsize=10)
axes[1, 0].set_title('RR Variability (Coefficient of Variation)', fontsize=12, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)
axes[1, 0].grid(axis='y', alpha=0.3)

axes[1, 1].bar(df_rr['record'], df_rr['rr_std'], color='#e87ba4', edgecolor='white', linewidth=0.5)
axes[1, 1].set_ylabel('RR Std Dev (ms)', fontsize=10)
axes[1, 1].set_title('RR Standard Deviation per Record', fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../rr_analysis.png', dpi=150, facecolor='#fcfcfb')
print("Saved rr_analysis.png")

# dashboard.png
fig = plt.figure(figsize=(20, 16))
per_rec = aami_data['per_record']
recs = [r['record'] for r in per_rec]

ax1 = plt.subplot(3, 4, 1)
classes = ['N', 'S', 'V', 'F', 'Q', 'nonbeat']
counts = [overall[c] for c in classes]
labels = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown', 'Non-beat']
colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
ax1.bar(labels, counts, color=colors, edgecolor='white', linewidth=1)
ax1.set_ylabel('Count')
ax1.set_title('AAMI Class Distribution', fontweight='bold', fontsize=10)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(axis='y', alpha=0.3)

ax2 = plt.subplot(3, 4, 2)
v_counts = [r['V'] for r in per_rec]
ax2.bar(recs, v_counts, color='#1baf7a', edgecolor='white', linewidth=0.5)
ax2.set_ylabel('VEB Count')
ax2.set_title('Ventricular Ectopic Beats per Record', fontweight='bold', fontsize=10)
ax2.tick_params(axis='x', rotation=90, labelsize=6)
ax2.grid(axis='y', alpha=0.3)

ax3 = plt.subplot(3, 4, 3)
s_counts = [r['S'] for r in per_rec]
ax3.bar(recs, s_counts, color='#eb6834', edgecolor='white', linewidth=0.5)
ax3.set_ylabel('SVEB Count')
ax3.set_title('Supraventricular Ectopic Beats per Record', fontweight='bold', fontsize=10)
ax3.tick_params(axis='x', rotation=90, labelsize=6)
ax3.grid(axis='y', alpha=0.3)

ax4 = plt.subplot(3, 4, 4)
lead_counts = df_stats['lead'].value_counts()
ax4.pie(lead_counts.values, labels=lead_counts.index, colors=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
ax4.set_title('Lead Distribution', fontweight='bold', fontsize=10)

ax5 = plt.subplot(3, 4, 5)
for lead in df_stats['lead'].unique():
    lead_data = df_stats[df_stats['lead'] == lead]
    ax5.scatter(lead_data['record'], lead_data['mean'], label=lead, alpha=0.7, s=20)
ax5.set_ylabel('Mean (mV)')
ax5.set_title('Signal Mean by Record and Lead', fontweight='bold', fontsize=10)
ax5.tick_params(axis='x', rotation=90, labelsize=6)
ax5.legend(fontsize=7)
ax5.grid(axis='y', alpha=0.3)

ax6 = plt.subplot(3, 4, 6)
for lead in df_stats['lead'].unique():
    lead_data = df_stats[df_stats['lead'] == lead]
    ax6.scatter(lead_data['record'], lead_data['std'], label=lead, alpha=0.7, s=20)
ax6.set_ylabel('Std Dev (mV)')
ax6.set_title('Signal Std Dev by Record and Lead', fontweight='bold', fontsize=10)
ax6.tick_params(axis='x', rotation=90, labelsize=6)
ax6.legend(fontsize=7)
ax6.grid(axis='y', alpha=0.3)

ax7 = plt.subplot(3, 4, 7)
ax7.hist(df_rr['hr_mean'], bins=20, color='#2a78d6', edgecolor='white', alpha=0.8)
ax7.set_xlabel('Heart Rate (bpm)')
ax7.set_ylabel('Frequency')
ax7.set_title('Heart Rate Distribution', fontweight='bold', fontsize=10)
ax7.grid(axis='y', alpha=0.3)

ax8 = plt.subplot(3, 4, 8)
cv = df_rr['rr_std'] / df_rr['rr_mean'] * 100
ax8.hist(cv, bins=20, color='#eb6834', edgecolor='white', alpha=0.8)
ax8.set_xlabel('CV of RR (%)')
ax8.set_ylabel('Frequency')
ax8.set_title('RR Variability Distribution', fontweight='bold', fontsize=10)
ax8.grid(axis='y', alpha=0.3)

ax9 = plt.subplot(3, 4, 9)
beat_counts = [r['N'] + r['S'] + r['V'] + r['F'] + r['Q'] for r in per_rec]
ax9.bar(recs, beat_counts, color='#2a78d6', edgecolor='white', linewidth=0.5)
ax9.set_ylabel('Total Beats')
ax9.set_title('Total Annotated Beats per Record', fontweight='bold', fontsize=10)
ax9.tick_params(axis='x', rotation=90, labelsize=6)
ax9.grid(axis='y', alpha=0.3)

ax10 = plt.subplot(3, 4, 10)
veb_pct = []
for r in per_rec:
    total = r['N'] + r['S'] + r['V'] + r['F'] + r['Q']
    veb_pct.append(r['V'] / total * 100 if total > 0 else 0)
ax10.bar(recs, veb_pct, color='#1baf7a', edgecolor='white', linewidth=0.5)
ax10.set_ylabel('VEB %')
ax10.set_title('VEB Percentage per Record', fontweight='bold', fontsize=10)
ax10.tick_params(axis='x', rotation=90, labelsize=6)
ax10.grid(axis='y', alpha=0.3)

ax11 = plt.subplot(3, 4, 11)
sveb_pct = []
for r in per_rec:
    total = r['N'] + r['S'] + r['V'] + r['F'] + r['Q']
    sveb_pct.append(r['S'] / total * 100 if total > 0 else 0)
ax11.bar(recs, sveb_pct, color='#eb6834', edgecolor='white', linewidth=0.5)
ax11.set_ylabel('SVEB %')
ax11.set_title('SVEB Percentage per Record', fontweight='bold', fontsize=10)
ax11.tick_params(axis='x', rotation=90, labelsize=6)
ax11.grid(axis='y', alpha=0.3)

ax12 = plt.subplot(3, 4, 12)
dominance = []
for r in per_rec:
    total = r['N'] + r['S'] + r['V'] + r['F'] + r['Q']
    if total > 0:
        v_pct = r['V'] / total * 100
        s_pct = r['S'] / total * 100
        if v_pct > 50:
            dominance.append('V-dominant')
        elif s_pct > 50:
            dominance.append('S-dominant')
        elif v_pct > 10:
            dominance.append('High V')
        elif s_pct > 10:
            dominance.append('High S')
        else:
            dominance.append('Normal')
    else:
        dominance.append('Unknown')

dom_counts = pd.Series(dominance).value_counts()
ax12.pie(dom_counts.values, labels=dom_counts.index,
         colors=['#1baf7a', '#eb6834', '#2a78d6', '#eda100', '#e87ba4'],
         autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
ax12.set_title('Record Class Dominance', fontweight='bold', fontsize=10)

plt.suptitle('MIT-BIH Arrhythmia Database - Comprehensive Exploration Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../dashboard.png', dpi=150, facecolor='#fcfcfb')
print("Saved dashboard.png")