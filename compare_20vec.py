import re
import json

with open('twin_20vec_results.json') as f:
    twin_data = json.load(f)

log_path = r'C:\Users\shind\.gemini\antigravity\brain\6dec11d9-d6d4-468f-8265-a262b817e5c4\.system_generated\tasks\task-2908.log'
with open(log_path) as f:
    log_text = f.read()

spk_pattern = re.findall(r'Note: SNN_HARDWARE_SPIKE_COUNTS: \[(\d+),(\d+),(\d+),(\d+)\] Class=(\d+)', log_text)

class_names = ['Normal (N)', 'Supra (S)', 'Ventricular (V)', 'Fusion (F)']

header = f"{'Vec':^4} | {'True Class':<15} | {'Twin Spikes [N,S,V,F]':^23} | {'RTL Spikes [N,S,V,F]':^23} | {'Twin':^5} | {'RTL':^5} | {'Match?':^6}"
print('=' * len(header))
print(header)
print('=' * len(header))

matches = 0
spike_matches = 0
for k in range(min(len(spk_pattern), len(twin_data))):
    t = twin_data[k]
    true_cls = class_names[t['true_label']]
    twin_spk = t['twin_spikes']
    rtl_spk = [int(x) for x in spk_pattern[k][:4]]
    twin_p = t['twin_pred']
    rtl_p = int(spk_pattern[k][4])
    
    match = (twin_p == rtl_p)
    spk_match = (twin_spk == rtl_spk)
    if match: matches += 1
    if spk_match: spike_matches += 1
    
    m_str = 'MATCH' if match else 'DIFF'
    s_str = '*' if spk_match else ' '
    print(f"{k:^4} | {true_cls:<15} | {str(twin_spk):^23} | {str(rtl_spk):^22}{s_str}| {twin_p:^5} | {rtl_p:^5} | {m_str:^6}")

print('=' * len(header))
print(f"Prediction Match Rate: {matches}/{len(spk_pattern)} ({matches/len(spk_pattern)*100:.1f}%)")
print(f"Exact Spike-by-Spike Matches: {spike_matches}/{len(spk_pattern)}")
