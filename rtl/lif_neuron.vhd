----------------------------------------------------------------------------------
-- Module: LIF Neuron (Leaky Integrate-and-Fire)
-- Project: ECG-SNN-FPGA
-- Description:
--   Multiplier-free LIF neuron for Spiking Neural Network.
--   Implements: V_mem = V_mem - (V_mem >> 4) + I_syn
--   Leak factor beta = 1 - 1/16 = 0.9375 (achieved via bit-shift, zero DSPs)
--   Soft reset: if spike, V_mem = V_mem - threshold
--
-- Ports:
--   clk, rst       : Clock and synchronous reset
--   en             : Process enable (driven by FSM)
--   i_syn          : Signed 16-bit synaptic current input (from accumulator)
--   spike_out      : 1-bit output spike
--   v_mem_out      : Current membrane potential (for debugging / output layer)
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity lif_neuron is
    generic (
        V_THRESHOLD : signed(15 downto 0) := x"0080"  -- Threshold = 0.5 in Q8.8 format (0.5 * 256 = 128 = 0x0080)
    );
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        en        : in  std_logic;
        i_syn     : in  signed(15 downto 0);   -- Synaptic current (Q8.8 fixed-point)
        spike_out : out std_logic;
        v_mem_out : out signed(15 downto 0)
    );
end lif_neuron;

architecture rtl of lif_neuron is

    signal v_mem      : signed(15 downto 0) := (others => '0');
    signal v_leaked   : signed(15 downto 0);
    signal v_new      : signed(15 downto 0);
    signal spike_flag : std_logic;

begin

    ---------------------------------------------------------------------------
    -- Step 1: LEAK — Subtract 1/16th of the membrane potential
    -- v_leaked = v_mem - (v_mem >>> 4)
    -- The arithmetic right shift (>>>) preserves the sign bit.
    -- This is equivalent to: v_leaked = v_mem * 0.9375
    -- No multiplier needed! Just a shift and a subtract.
    ---------------------------------------------------------------------------
    v_leaked <= v_mem - shift_right(v_mem, 4);

    ---------------------------------------------------------------------------
    -- Step 2: INTEGRATE — Add synaptic current to leaked voltage
    -- v_new = v_leaked + i_syn
    ---------------------------------------------------------------------------
    v_new <= v_leaked + i_syn;

    ---------------------------------------------------------------------------
    -- Step 3: FIRE — Compare against threshold
    ---------------------------------------------------------------------------
    spike_flag <= '1' when v_new >= V_THRESHOLD else '0';

    ---------------------------------------------------------------------------
    -- Step 4: UPDATE — Register the new membrane potential
    --   If spike fired: soft reset (subtract threshold)
    --   If no spike: just store the new value
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                v_mem <= (others => '0');
            elsif en = '1' then
                if spike_flag = '1' then
                    -- Soft Reset: keep the leftover energy above threshold
                    v_mem <= v_new - V_THRESHOLD;
                else
                    v_mem <= v_new;
                end if;
            end if;
        end if;
    end process;

    ---------------------------------------------------------------------------
    -- Output assignments
    ---------------------------------------------------------------------------
    spike_out <= spike_flag when en = '1' else '0';
    v_mem_out <= v_mem;

end rtl;
