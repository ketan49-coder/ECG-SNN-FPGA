import json
import numpy as np
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0")

# Load all data
with open('../aami_distribution.json', 'r') as f:
    aami_data = json.load(f)

df_stats = pd.read_csv('../signal_stats.csv')
df_rr = pd.read_csv('../rr_intervals.csv')

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 16))

# 1. AAMI Distribution
ax1 = plt.subplot(3, 4, 1)
overall = aami_data['overall']
classes = ['N', 'S', 'V', 'F', 'Q', 'nonbeat']
counts = [overall[c] for c in classes]
labels = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown', 'Non-beat']
colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
bars = ax1.bar(labels, counts, color=colors, edgecolor='white', linewidth=1)
ax1.set_ylabel('Count')
ax1.set_title('AAMI Class Distribution', fontweight='bold', fontsize=10)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(axis='y', alpha=0.3)

# 2. Per-record VEB count
ax2 = plt.subplot(3, 4, 2)
per_rec = aami_data['per_record']
recs = [r['record'] for r in per_rec]
v_counts = [r['V'] for r in per_rec]
ax2.bar(recs, v_counts, color='#1baf7a', edgecolor='white', linewidth=0.5)
ax2.set_ylabel('VEB Count')
ax2.set_title('Ventricular Ectopic Beats per Record', fontweight='bold', fontsize=10)
ax2.tick_params(axis='x', rotation=90, labelsize=6)
ax2.grid(axis='y', alpha=0.3)

# 3. Per-record SVEB count
ax3 = plt.subplot(3, 4, 3)
s_counts = [r['S'] for r in per_rec]
ax3.bar(recs, s_counts, color='#eb6834', edgecolor='white', linewidth=0.5)
ax3.set_ylabel('SVEB Count')
ax3.set_title('Supraventricular Ectopic Beats per Record', fontweight='bold', fontsize=10)
ax3.tick_params(axis='x', rotation=90, labelsize=6)
ax3.grid(axis='y', alpha=0.3)

# 4. Lead distribution
ax4 = plt.subplot(3, 4, 4)
lead_counts = df_stats['lead'].value_counts()
ax4.pie(lead_counts.values, labels=lead_counts.index, colors=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
ax4.set_title('Lead Distribution', fontweight='bold', fontsize=10)

# 5. Signal mean by lead
ax5 = plt.subplot(3, 4, 5)
for lead in df_stats['lead'].unique():
    lead_data = df_stats[df_stats['lead'] == lead]
    ax5.scatter(lead_data['record'], lead_data['mean'], label=lead, alpha=0.7, s=20)
ax5.set_ylabel('Mean (mV)')
ax5.set_title('Signal Mean by Record and Lead', fontweight='bold', fontsize=10)
ax5.tick_params(axis='x', rotation=90, labelsize=6)
ax5.legend(fontsize=7)
ax5.grid(axis='y', alpha=0.3)

# 6. Signal std by lead
ax6 = plt.subplot(3, 4, 6)
for lead in df_stats['lead'].unique():
    lead_data = df_stats[df_stats['lead'] == lead]
    ax6.scatter(lead_data['record'], lead_data['std'], label=lead, alpha=0.7, s=20)
ax6.set_ylabel('Std Dev (mV)')
ax6.set_title('Signal Std Dev by Record and Lead', fontweight='bold', fontsize=10)
ax6.tick_params(axis='x', rotation=90, labelsize=6)
ax6.legend(fontsize=7)
ax6.grid(axis='y', alpha=0.3)

# 7. Heart rate distribution
ax7 = plt.subplot(3, 4, 7)
ax7.hist(df_rr['hr_mean'], bins=20, color='#2a78d6', edgecolor='white', alpha=0.8)
ax7.set_xlabel('Heart Rate (bpm)')
ax7.set_ylabel('Frequency')
ax7.set_title('Heart Rate Distribution', fontweight='bold', fontsize=10)
ax7.grid(axis='y', alpha=0.3)

# 8. RR variability
ax8 = plt.subplot(3, 4, 8)
cv = df_rr['rr_std'] / df_rr['rr_mean'] * 100
ax8.hist(cv, bins=20, color='#eb6834', edgecolor='white', alpha=0.8)
ax8.set_xlabel('CV of RR (%)')
ax8.set_ylabel('Frequency')
ax8.set_title('RR Variability Distribution', fontweight='bold', fontsize=10)
ax8.grid(axis='y', alpha=0.3)

# 9. Beat count per record
ax9 = plt.subplot(3, 4, 9)
beat_counts = [r['N'] + r['S'] + r['V'] + r['F'] + r['Q'] for r in per_rec]
ax9.bar(recs, beat_counts, color='#2a78d6', edgecolor='white', linewidth=0.5)
ax9.set_ylabel('Total Beats')
ax9.set_title('Total Annotated Beats per Record', fontweight='bold', fontsize=10)
ax9.tick_params(axis='x', rotation=90, labelsize=6)
ax9.grid(axis='y', alpha=0.3)

# 10. VEB % per record
ax10 = plt.subplot(3, 4, 10)
veb_pct = []
for r in per_rec:
    total = r['N'] + r['S'] + r['V'] + r['F'] + r['Q']
    if total > 0:
        veb_pct.append(r['V'] / total * 100)
    else:
        veb_pct.append(0)
ax10.bar(recs, veb_pct, color='#1baf7a', edgecolor='white', linewidth=0.5)
ax10.set_ylabel('VEB %')
ax10.set_title('VEB Percentage per Record', fontweight='bold', fontsize=10)
ax10.tick_params(axis='x', rotation=90, labelsize=6)
ax10.grid(axis='y', alpha=0.3)

# 11. SVEB % per record
ax11 = plt.subplot(3, 4, 11)
sveb_pct = []
for r in per_rec:
    total = r['N'] + r['S'] + r['V'] + r['F'] + r['Q']
    if total > 0:
        sveb_pct.append(r['S'] / total * 100)
    else:
        sveb_pct.append(0)
ax11.bar(recs, sveb_pct, color='#eb6834', edgecolor='white', linewidth=0.5)
ax11.set_ylabel('SVEB %')
ax11.set_title('SVEB Percentage per Record', fontweight='bold', fontsize=10)
ax11.tick_params(axis='x', rotation=90, labelsize=6)
ax11.grid(axis='y', alpha=0.3)

# 12. Records by class dominance
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

# Summary
print("\n=== DATASET SUMMARY ===")
print("Total Records:", len(per_rec))
print("Total Duration: {:.1f} minutes ({:.1f} hours)".format(len(per_rec) * 30.1, len(per_rec) * 30.1 / 60))
print("Total Beats: {:,}".format(sum(beat_counts)))
print("Sampling Rate: 360 Hz")
print("Leads:", df_stats['lead'].unique().tolist())
print("\nAAMI Class Distribution:")
for c in ['N', 'S', 'V', 'F', 'Q']:
    pct = overall[c] / sum(overall.values()) * 100
    print("  {}: {:,} ({:.1f}%)".format(c, overall[c], pct))
print("  Non-beat: {:,} ({:.1f}%)".format(overall['nonbeat'], overall['nonbeat']/sum(overall.values())*100))