import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load Data
d = np.load('mitbih_sparrow_4class_128.npz')
X_all, y_all = d['X'], d['y']
X_tv, X_test, y_tv, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)
X_test_8bit = np.clip(np.round(X_test * 255.0), 0, 255).astype(np.int32)

# Pick 5 of each class (N=0, S=1, V=2, F=3)
vec_indices = []
for c in range(4):
    idx = np.where(y_test == c)[0][:5]
    vec_indices.extend(idx)

vec_indices = np.array(vec_indices)
samples_20 = X_test_8bit[vec_indices]
labels_20 = y_test[vec_indices]

# 2. Run Python Twin to get reference spike counts
w = np.load('weights.npz')
W1, W2, W3 = w['W1'].astype(np.int32), w['W2'].astype(np.int32), w['W3'].astype(np.int32)

class HardwareLFSR8:
    def __init__(self, seed=0xA5):
        self.state = seed & 0xFF
        if self.state == 0: self.state = 0xA5
    def peek(self):
        return self.state
    def step(self):
        bit = ((self.state >> 7) ^ (self.state >> 5) ^ (self.state >> 4) ^ (self.state >> 3)) & 1
        self.state = ((self.state << 1) | bit) & 0xFF
        return self.state

# In hardware, LFSR runs continuously across timesteps. Let's see twin with continuous LFSR vs reset LFSR
twin_results = []
lfsr_cont = HardwareLFSR8(0xA5)
for k in range(20):
    sample = samples_20[k]
    v1 = np.zeros(32, dtype=np.int32)
    v2 = np.zeros(32, dtype=np.int32)
    v3 = np.zeros(4,  dtype=np.int32)
    spike_counts = np.zeros(4, dtype=np.int32)
    for t in range(75):
        spk_in = np.zeros(128, dtype=np.int32)
        for i in range(128):
            if sample[i] > lfsr_cont.peek():
                spk_in[i] = 1
            lfsr_cont.step()
        v1 = v1 - np.right_shift(v1, 4) + np.dot(W1, spk_in)
        s1 = (v1 >= 128).astype(np.int32); v1 = np.where(s1 == 1, 0, v1)
        v2 = v2 - np.right_shift(v2, 4) + np.dot(W2, s1)
        s2 = (v2 >= 128).astype(np.int32); v2 = np.where(s2 == 1, 0, v2)
        v3 = v3 - np.right_shift(v3, 4) + np.dot(W3, s2)
        s3 = (v3 >= 128).astype(np.int32); v3 = np.where(s3 == 1, 0, v3)
        spike_counts += s3
    pred = np.argmax(spike_counts)
    twin_results.append({
        'index': k,
        'true_label': int(labels_20[k]),
        'twin_spikes': spike_counts.tolist(),
        'twin_pred': int(pred)
    })

import json
with open('twin_20vec_results.json', 'w') as f:
    json.dump(twin_results, f, indent=2)

print('Computed Python twin reference for 20 vectors!')

# 3. Format VHDL Testbench
def format_samples_vhdl(s):
    lines = []
    for i in range(0, 128, 16):
        chunk = s[i:i+16]
        lines.append('            ' + ', '.join([f'{x:>3}' for x in chunk]))
    return ',\n'.join(lines)

vectors_vhdl = []
for k in range(20):
    vectors_vhdl.append(f'        -- Vector {k} (Class {labels_20[k]})\n        (\n{format_samples_vhdl(samples_20[k])}\n        )')

vectors_vhdl_str = ',\n'.join(vectors_vhdl)
labels_vhdl_str = ', '.join([str(x) for x in labels_20])

tb_vhdl = f"""----------------------------------------------------------------------------------
-- Testbench: snn_top_tb (20-Vector Comprehensive Hardware Verification)
-- Project: ECG-SNN-FPGA (Spartan-7 xc7s15)
-- Evaluates 20 real patient beats across all 4 classes (5 N, 5 S, 5 V, 5 F)
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_pkg.ALL;

entity snn_top_tb is
end snn_top_tb;

architecture behavior of snn_top_tb is

    component snn_top
        port (
            clk       : in  std_logic;
            rst       : in  std_logic;
            data_in   : in  std_logic_vector(7 downto 0);
            addr_in   : in  std_logic_vector(6 downto 0);
            wr_en     : in  std_logic;
            start     : in  std_logic;
            class_out : out std_logic_vector(2 downto 0);
            done      : out std_logic;
            busy      : out std_logic
        );
    end component;

    signal clk       : std_logic := '0';
    signal rst       : std_logic := '0';
    signal data_in   : std_logic_vector(7 downto 0) := (others => '0');
    signal addr_in   : std_logic_vector(6 downto 0) := (others => '0');
    signal wr_en     : std_logic := '0';
    signal start     : std_logic := '0';
    signal class_out : std_logic_vector(2 downto 0);
    signal done      : std_logic;
    signal busy      : std_logic;

    constant clk_period : time := 10 ns;

    type ecg_sample_array_t is array (0 to 127) of integer range 0 to 255;
    type test_vectors_t is array (0 to 19) of ecg_sample_array_t;
    type labels_t is array (0 to 19) of integer range 0 to 3;

    constant TEST_VECTORS : test_vectors_t := (
{vectors_vhdl_str}
    );

    constant TEST_LABELS : labels_t := ({labels_vhdl_str});

begin

    uut: snn_top
        port map (
            clk       => clk,
            rst       => rst,
            data_in   => data_in,
            addr_in   => addr_in,
            wr_en     => wr_en,
            start     => start,
            class_out => class_out,
            done      => done,
            busy      => busy
        );

    clk_process: process
    begin
        clk <= '0';
        wait for clk_period / 2;
        clk <= '1';
        wait for clk_period / 2;
    end process;

    stim_proc: process
    begin
        report "==========================================================" severity note;
        report "   STARTING 20-VECTOR SNN HARDWARE SIMULATION (T = 75)    " severity note;
        report "==========================================================" severity note;

        rst   <= '1';
        start <= '0';
        wr_en <= '0';
        wait for 100 ns;
        rst   <= '0';
        wait for clk_period * 5;

        for v_idx in 0 to 19 loop
            -- Load 128 samples
            for i in 0 to 127 loop
                addr_in <= std_logic_vector(to_unsigned(i, 7));
                data_in <= std_logic_vector(to_unsigned(TEST_VECTORS(v_idx)(i), 8));
                wr_en   <= '1';
                wait for clk_period;
            end loop;
            wr_en <= '0';
            wait for clk_period * 2;

            -- Trigger Inference
            start <= '1';
            wait for clk_period;
            start <= '0';

            wait until done = '1';
            wait for clk_period * 2;

            report "RTL_VEC_" & integer'image(v_idx) & 
                   "_TRUE_" & integer'image(TEST_LABELS(v_idx)) & 
                   "_PRED_" & integer'image(to_integer(unsigned(class_out))) severity note;

            wait for clk_period * 10;
        end loop;

        report "==========================================================" severity note;
        report ">> 20-VECTOR SIMULATION COMPLETE!                         " severity note;
        report "==========================================================" severity note;

        wait;
    end process;

end behavior;
"""

with open('rtl/snn_top_tb_20vec.vhd', 'w') as f:
    f.write(tb_vhdl)

with open('SNN-FPGA-ECG/SNN-FPGA-ECG.srcs/sim_1/imports/rtl/snn_top_tb.vhd', 'w') as f:
    f.write(tb_vhdl)

print("Created 20-vector testbench and updated Vivado sim_1 files!")
