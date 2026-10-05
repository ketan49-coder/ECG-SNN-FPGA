----------------------------------------------------------------------------------
-- Package: SNN Types, Constants, and Weight Arrays
-- Project: ECG-SNN-FPGA
-- Description:
--   Central package defining all network parameters, fixed-point types,
--   and weight storage arrays. The weight arrays are initialized to zero
--   as placeholders — run export_weights.py to regenerate this file with
--   trained, quantized weights from the Python model.
--
-- Fixed-Point Format: Q8.8 (8 integer bits, 8 fractional bits = 16-bit signed)
-- Weight Format: 8-bit signed integer (gets sign-extended to Q8.8 on use)
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package snn_pkg is

    ---------------------------------------------------------------------------
    -- Network Architecture Constants
    ---------------------------------------------------------------------------
    constant NUM_INPUTS    : integer := 128;   -- ECG window samples
    constant NUM_HIDDEN_1  : integer := 16;    -- Hidden layer 1 neurons
    constant NUM_HIDDEN_2  : integer := 16;    -- Hidden layer 2 neurons
    constant NUM_OUTPUTS   : integer := 5;     -- Output classes (N, S, V, F, Q)
    constant NUM_TIMESTEPS : integer := 16;    -- SNN simulation timesteps

    ---------------------------------------------------------------------------
    -- LIF Neuron Parameters
    ---------------------------------------------------------------------------
    constant V_THRESHOLD : signed(15 downto 0) := x"0080";  -- 0.5 in Q8.8
    constant LEAK_SHIFT  : integer := 4;  -- beta = 1 - 1/2^4 = 0.9375

    ---------------------------------------------------------------------------
    -- Array Type Definitions
    ---------------------------------------------------------------------------
    -- Input register file
    type input_array_t is array (0 to NUM_INPUTS - 1) of unsigned(7 downto 0);

    -- Membrane potential arrays
    type vmem_array_16_t is array (0 to 15) of signed(15 downto 0);
    type vmem_array_5_t  is array (0 to 4)  of signed(15 downto 0);

    -- Spike counter array for output layer
    type spike_count_5_t is array (0 to 4) of unsigned(7 downto 0);

    -- Weight arrays (flattened 2D: index = neuron * num_inputs + input)
    type weight_array_h1_t  is array (0 to NUM_INPUTS   * NUM_HIDDEN_1 - 1) of signed(7 downto 0);
    type weight_array_h2_t  is array (0 to NUM_HIDDEN_1  * NUM_HIDDEN_2 - 1) of signed(7 downto 0);
    type weight_array_out_t is array (0 to NUM_HIDDEN_2  * NUM_OUTPUTS  - 1) of signed(7 downto 0);

    ---------------------------------------------------------------------------
    -- Placeholder Weight Arrays (replaced by export_weights.py after training)
    ---------------------------------------------------------------------------
    -- Layer 1: 128 inputs x 16 neurons = 2048 weights
    constant WEIGHTS_H1 : weight_array_h1_t := (others => (others => '0'));

    -- Layer 2: 16 inputs x 16 neurons = 256 weights
    constant WEIGHTS_H2 : weight_array_h2_t := (others => (others => '0'));

    -- Output: 16 inputs x 5 neurons = 80 weights
    constant WEIGHTS_OUT : weight_array_out_t := (others => (others => '0'));

end package snn_pkg;
