----------------------------------------------------------------------------------
-- Module: SNN Top-Level
-- Project: ECG-SNN-FPGA
-- Description:
--   Complete Spiking Neural Network for ECG arrhythmia classification.
--   Architecture: 128 inputs -> 16 hidden -> 16 hidden -> 5 outputs
--
--   Data Flow:
--     1. External system loads 128 normalized samples via data_in/addr_in/wr_en
--     2. Assert 'start' to begin classification
--     3. FSM runs 16 timesteps of: Encode -> H1 -> H2 -> Output
--     4. Argmax selects winning class, asserts 'done'
--
--   Processing: Sequential (one neuron at a time, one input at a time)
--   Total cycles per heartbeat: ~16 * (128 + 16*129 + 16*17 + 5*17) ≈ 40,000
--   At 100 MHz: ~0.4 ms per classification (2,500 heartbeats/second)
--
--   Resource Usage Target:
--     DSP Slices: 0 (multiplier-free design)
--     BRAM: 1-2 blocks for weight storage
--     LUTs: ~2,000-4,000
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_pkg.ALL;

entity snn_top is
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        -- Input interface: load one sample at a time
        data_in   : in  std_logic_vector(7 downto 0);   -- 8-bit normalized sample
        addr_in   : in  std_logic_vector(6 downto 0);   -- Address 0..127
        wr_en     : in  std_logic;                       -- Write enable
        start     : in  std_logic;                       -- Begin classification
        -- Output interface
        class_out : out std_logic_vector(2 downto 0);    -- Predicted class (0..4)
        done      : out std_logic;                       -- Classification complete
        busy      : out std_logic                        -- Processing in progress
    );
end snn_top;

architecture rtl of snn_top is

    ---------------------------------------------------------------------------
    -- FSM State Type
    ---------------------------------------------------------------------------
    type state_t is (
        S_IDLE,          -- Waiting for start signal
        S_ENCODE,        -- Rate encoding (128 cycles per timestep)
        S_PROC_H1,       -- Processing hidden layer 1
        S_PROC_H2,       -- Processing hidden layer 2
        S_PROC_OUT,      -- Processing output layer
        S_CHECK_TS,      -- Check timestep counter
        S_CLASSIFY,      -- Run argmax on output spike counts
        S_DONE           -- Output result, wait for acknowledge
    );
    signal state : state_t := S_IDLE;

    ---------------------------------------------------------------------------
    -- Input Register File (128 x 8-bit)
    ---------------------------------------------------------------------------
    signal input_regs : input_array_t := (others => (others => '0'));

    ---------------------------------------------------------------------------
    -- Spike Vectors
    ---------------------------------------------------------------------------
    signal spikes_encoded : std_logic_vector(NUM_INPUTS - 1 downto 0)   := (others => '0');
    signal spikes_h1      : std_logic_vector(NUM_HIDDEN_1 - 1 downto 0) := (others => '0');
    signal spikes_h2      : std_logic_vector(NUM_HIDDEN_2 - 1 downto 0) := (others => '0');

    ---------------------------------------------------------------------------
    -- Membrane Potential Registers
    ---------------------------------------------------------------------------
    signal vmem_h1  : vmem_array_16_t := (others => (others => '0'));
    signal vmem_h2  : vmem_array_16_t := (others => (others => '0'));
    signal vmem_out : vmem_array_5_t  := (others => (others => '0'));

    ---------------------------------------------------------------------------
    -- Output Spike Counters (sum spikes across all 16 timesteps)
    ---------------------------------------------------------------------------
    signal spike_count_out : spike_count_5_t := (others => (others => '0'));

    ---------------------------------------------------------------------------
    -- Processing Counters
    ---------------------------------------------------------------------------
    signal neuron_idx : unsigned(4 downto 0) := (others => '0');  -- 0..31
    signal input_idx  : unsigned(7 downto 0) := (others => '0');  -- 0..255
    signal timestep   : unsigned(4 downto 0) := (others => '0');  -- 0..31

    ---------------------------------------------------------------------------
    -- Synaptic Current Accumulator
    ---------------------------------------------------------------------------
    signal accum : signed(15 downto 0) := (others => '0');

    ---------------------------------------------------------------------------
    -- LFSR Signals
    ---------------------------------------------------------------------------
    signal lfsr_en  : std_logic := '0';
    signal lfsr_out : std_logic_vector(7 downto 0);

    ---------------------------------------------------------------------------
    -- Argmax Signals
    ---------------------------------------------------------------------------
    signal argmax_result : std_logic_vector(2 downto 0);

    ---------------------------------------------------------------------------
    -- Internal output registers
    ---------------------------------------------------------------------------
    signal done_reg  : std_logic := '0';
    signal busy_reg  : std_logic := '0';
    signal class_reg : std_logic_vector(2 downto 0) := "000";

begin

    ---------------------------------------------------------------------------
    -- Component Instantiations
    ---------------------------------------------------------------------------

    -- LFSR for pseudo-random rate encoding
    lfsr_inst : entity work.lfsr_8bit
        port map (
            clk      => clk,
            rst      => rst,
            en       => lfsr_en,
            lfsr_out => lfsr_out
        );

    -- Argmax classifier (combinational — always computes the winning class)
    -- Uses spike counts sign-extended to 16-bit for comparison
    argmax_inst : entity work.argmax_5
        port map (
            val_0     => signed(resize(spike_count_out(0), 16)),
            val_1     => signed(resize(spike_count_out(1), 16)),
            val_2     => signed(resize(spike_count_out(2), 16)),
            val_3     => signed(resize(spike_count_out(3), 16)),
            val_4     => signed(resize(spike_count_out(4), 16)),
            class_out => argmax_result
        );

    ---------------------------------------------------------------------------
    -- Input Loading Process (active only during S_IDLE)
    ---------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if wr_en = '1' and state = S_IDLE then
                input_regs(to_integer(unsigned(addr_in))) <= unsigned(data_in);
            end if;
        end if;
    end process;

    ---------------------------------------------------------------------------
    -- Main FSM and Datapath Process
    ---------------------------------------------------------------------------
    process(clk)
        variable v_leaked : signed(15 downto 0);
        variable v_new    : signed(15 downto 0);
        variable w_addr   : integer range 0 to 2047;
        variable w_val    : signed(7 downto 0);
        variable w_ext    : signed(15 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                -- Global reset
                state          <= S_IDLE;
                neuron_idx     <= (others => '0');
                input_idx      <= (others => '0');
                timestep       <= (others => '0');
                accum          <= (others => '0');
                spikes_encoded <= (others => '0');
                spikes_h1      <= (others => '0');
                spikes_h2      <= (others => '0');
                vmem_h1        <= (others => (others => '0'));
                vmem_h2        <= (others => (others => '0'));
                vmem_out       <= (others => (others => '0'));
                spike_count_out <= (others => (others => '0'));
                lfsr_en        <= '0';
                done_reg       <= '0';
                busy_reg       <= '0';
                class_reg      <= "000";
            else
                -- Defaults each cycle
                lfsr_en  <= '0';
                done_reg <= '0';

                case state is

                    -------------------------------------------------------
                    -- S_IDLE: Wait for external start signal
                    -------------------------------------------------------
                    when S_IDLE =>
                        busy_reg <= '0';
                        if start = '1' then
                            -- Reset all state for a new heartbeat classification
                            vmem_h1         <= (others => (others => '0'));
                            vmem_h2         <= (others => (others => '0'));
                            vmem_out        <= (others => (others => '0'));
                            spike_count_out <= (others => (others => '0'));
                            spikes_h1       <= (others => '0');
                            spikes_h2       <= (others => '0');
                            timestep        <= (others => '0');
                            input_idx       <= (others => '0');
                            busy_reg        <= '1';
                            state           <= S_ENCODE;
                        end if;

                    -------------------------------------------------------
                    -- S_ENCODE: Rate encode all 128 inputs (128 cycles)
                    --   For each input: if input_val > LFSR → spike = 1
                    --   Higher amplitude → higher probability of spiking
                    -------------------------------------------------------
                    when S_ENCODE =>
                        lfsr_en <= '1';  -- Advance LFSR each cycle

                        if input_regs(to_integer(input_idx)) > unsigned(lfsr_out) then
                            spikes_encoded(to_integer(input_idx)) <= '1';
                        else
                            spikes_encoded(to_integer(input_idx)) <= '0';
                        end if;

                        if input_idx = NUM_INPUTS - 1 then
                            -- All 128 inputs encoded, move to hidden layer 1
                            input_idx  <= (others => '0');
                            neuron_idx <= (others => '0');
                            accum      <= (others => '0');
                            state      <= S_PROC_H1;
                        else
                            input_idx <= input_idx + 1;
                        end if;

                    -------------------------------------------------------
                    -- S_PROC_H1: Process Hidden Layer 1
                    --   16 neurons, each with 128 inputs
                    --   Sequential: accumulate weighted spikes, then LIF
                    -------------------------------------------------------
                    when S_PROC_H1 =>
                        if input_idx < NUM_INPUTS then
                            -- ACCUMULATE PHASE: add weight if spike is present
                            if spikes_encoded(to_integer(input_idx)) = '1' then
                                w_addr := to_integer(neuron_idx) * NUM_INPUTS
                                          + to_integer(input_idx);
                                w_val  := WEIGHTS_H1(w_addr);
                                w_ext  := resize(w_val, 16);
                                accum  <= accum + w_ext;
                            end if;
                            input_idx <= input_idx + 1;
                        else
                            -- LIF UPDATE PHASE
                            -- Step 1: Leak
                            v_leaked := vmem_h1(to_integer(neuron_idx))
                                        - shift_right(vmem_h1(to_integer(neuron_idx)), LEAK_SHIFT);
                            -- Step 2: Integrate
                            v_new := v_leaked + accum;
                            -- Step 3: Fire & Reset
                            if v_new >= V_THRESHOLD then
                                vmem_h1(to_integer(neuron_idx)) <= (others => '0');  -- Zero reset
                                spikes_h1(to_integer(neuron_idx)) <= '1';
                            else
                                vmem_h1(to_integer(neuron_idx)) <= v_new;
                                spikes_h1(to_integer(neuron_idx)) <= '0';
                            end if;

                            -- Prepare for next neuron
                            accum     <= (others => '0');
                            input_idx <= (others => '0');

                            if neuron_idx = NUM_HIDDEN_1 - 1 then
                                neuron_idx <= (others => '0');
                                state      <= S_PROC_H2;
                            else
                                neuron_idx <= neuron_idx + 1;
                            end if;
                        end if;

                    -------------------------------------------------------
                    -- S_PROC_H2: Process Hidden Layer 2
                    --   16 neurons, each with 16 inputs (from H1 spikes)
                    -------------------------------------------------------
                    when S_PROC_H2 =>
                        if input_idx < NUM_HIDDEN_1 then
                            -- ACCUMULATE PHASE
                            if spikes_h1(to_integer(input_idx)) = '1' then
                                w_addr := to_integer(neuron_idx) * NUM_HIDDEN_1
                                          + to_integer(input_idx);
                                w_val  := WEIGHTS_H2(w_addr);
                                w_ext  := resize(w_val, 16);
                                accum  <= accum + w_ext;
                            end if;
                            input_idx <= input_idx + 1;
                        else
                            -- LIF UPDATE PHASE
                            v_leaked := vmem_h2(to_integer(neuron_idx))
                                        - shift_right(vmem_h2(to_integer(neuron_idx)), LEAK_SHIFT);
                            v_new := v_leaked + accum;

                            if v_new >= V_THRESHOLD then
                                vmem_h2(to_integer(neuron_idx)) <= (others => '0');
                                spikes_h2(to_integer(neuron_idx)) <= '1';
                            else
                                vmem_h2(to_integer(neuron_idx)) <= v_new;
                                spikes_h2(to_integer(neuron_idx)) <= '0';
                            end if;

                            accum     <= (others => '0');
                            input_idx <= (others => '0');

                            if neuron_idx = NUM_HIDDEN_2 - 1 then
                                neuron_idx <= (others => '0');
                                state      <= S_PROC_OUT;
                            else
                                neuron_idx <= neuron_idx + 1;
                            end if;
                        end if;

                    -------------------------------------------------------
                    -- S_PROC_OUT: Process Output Layer
                    --   5 neurons, each with 16 inputs (from H2 spikes)
                    --   Tracks spike counts across all 16 timesteps
                    -------------------------------------------------------
                    when S_PROC_OUT =>
                        if input_idx < NUM_HIDDEN_2 then
                            -- ACCUMULATE PHASE
                            if spikes_h2(to_integer(input_idx)) = '1' then
                                w_addr := to_integer(neuron_idx) * NUM_HIDDEN_2
                                          + to_integer(input_idx);
                                w_val  := WEIGHTS_OUT(w_addr);
                                w_ext  := resize(w_val, 16);
                                accum  <= accum + w_ext;
                            end if;
                            input_idx <= input_idx + 1;
                        else
                            -- LIF UPDATE PHASE
                            v_leaked := vmem_out(to_integer(neuron_idx))
                                        - shift_right(vmem_out(to_integer(neuron_idx)), LEAK_SHIFT);
                            v_new := v_leaked + accum;

                            if v_new >= V_THRESHOLD then
                                vmem_out(to_integer(neuron_idx)) <= (others => '0');
                                -- Increment spike counter for this output neuron
                                spike_count_out(to_integer(neuron_idx)) <=
                                    spike_count_out(to_integer(neuron_idx)) + 1;
                            else
                                vmem_out(to_integer(neuron_idx)) <= v_new;
                            end if;

                            accum     <= (others => '0');
                            input_idx <= (others => '0');

                            if neuron_idx = NUM_OUTPUTS - 1 then
                                neuron_idx <= (others => '0');
                                state      <= S_CHECK_TS;
                            else
                                neuron_idx <= neuron_idx + 1;
                            end if;
                        end if;

                    -------------------------------------------------------
                    -- S_CHECK_TS: Check if all 16 timesteps are complete
                    -------------------------------------------------------
                    when S_CHECK_TS =>
                        if timestep = NUM_TIMESTEPS - 1 then
                            -- All timesteps done → classify
                            state <= S_CLASSIFY;
                        else
                            -- More timesteps to process
                            timestep  <= timestep + 1;
                            input_idx <= (others => '0');
                            -- Spikes are regenerated each timestep
                            -- Membrane potentials persist (temporal memory!)
                            state <= S_ENCODE;
                        end if;

                    -------------------------------------------------------
                    -- S_CLASSIFY: Latch the argmax result
                    -------------------------------------------------------
                    when S_CLASSIFY =>
                        class_reg <= argmax_result;
                        state     <= S_DONE;

                    -------------------------------------------------------
                    -- S_DONE: Output the classification result
                    -------------------------------------------------------
                    when S_DONE =>
                        done_reg <= '1';
                        busy_reg <= '0';
                        if start = '0' then
                            state <= S_IDLE;
                        end if;

                    when others =>
                        state <= S_IDLE;

                end case;
            end if;
        end if;
    end process;

    ---------------------------------------------------------------------------
    -- Output Port Assignments
    ---------------------------------------------------------------------------
    class_out <= class_reg;
    done      <= done_reg;
    busy      <= busy_reg;

end rtl;
