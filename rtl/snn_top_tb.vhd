----------------------------------------------------------------------------------
-- Testbench: snn_top_tb
-- Project: ECG-SNN-FPGA (Spartan-7 xc7s15)
-- Description:
--   Cycle-accurate behavioral simulation testbench for 4-class ECG SNN.
--   Validates hardware inference using real patient ECG heartbeats:
--     Test 1: Normal Sinus Beat (N) -> Expected: Class 0 ("000")
--     Test 2: Premature Ventricular Ectopic Beat (V) -> Expected: Class 2 ("010")
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_pkg.ALL;

entity snn_top_tb is
    -- Testbench has no external ports
end snn_top_tb;

architecture behavior of snn_top_tb is

    -- Component Declaration for Unit Under Test (UUT)
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

    -- UUT Signals
    signal clk       : std_logic := '0';
    signal rst       : std_logic := '0';
    signal data_in   : std_logic_vector(7 downto 0) := (others => '0');
    signal addr_in   : std_logic_vector(6 downto 0) := (others => '0');
    signal wr_en     : std_logic := '0';
    signal start     : std_logic := '0';
    signal class_out : std_logic_vector(2 downto 0);
    signal done      : std_logic;
    signal busy      : std_logic;

    -- Clock period: 100 MHz oscillator = 10 ns
    constant clk_period : time := 10 ns;

    -- Sample array type for 128-point ECG window
    type ecg_window_t is array (0 to 127) of integer range 0 to 255;

    ---------------------------------------------------------------------------
    -- Real Patient Test Vector 1: Normal Sinus Rhythm (N)
    -- True Label: Normal (Class 0)
    ---------------------------------------------------------------------------
    constant PATIENT_BEAT_N : ecg_window_t := (
        71, 70, 69, 69, 69, 74, 75, 75, 74, 75,
        77, 79, 80, 80, 81, 82, 83, 86, 86, 85,
        84, 84, 84, 83, 83, 80, 78, 75, 74, 75,
        75, 74, 72, 71, 72, 72, 73, 71, 70, 69,
        70, 71, 71, 71, 70, 70, 70, 72, 72, 71,
        70, 71, 71, 71, 70, 70, 73, 79, 91, 107,
        128, 154, 179, 202, 219, 221, 201, 170, 134, 102,
        80, 73, 70, 70, 69, 68, 69, 73, 74, 72,
        70, 69, 69, 71, 71, 70, 69, 69, 71, 71,
        73, 72, 71, 70, 71, 72, 72, 72, 72, 72,
        72, 74, 74, 72, 73, 73, 75, 76, 78, 78,
        77, 76, 79, 81, 81, 80, 79, 79, 82, 84,
        85, 84, 84, 85, 86, 89, 90, 91
    );

    ---------------------------------------------------------------------------
    -- Real Patient Test Vector 2: Premature Ventricular Contraction (V)
    -- True Label: Ventricular (Class 2)
    ---------------------------------------------------------------------------
    constant PATIENT_BEAT_V : ecg_window_t := (
        112, 113, 112, 109, 107, 106, 105, 106, 105, 103,
        100,  98,  98,  97,  96,  96,  94,  93,  95,  96,
         94,  92,  91,  91,  92,  94,  92,  93,  93,  93,
         95,  97,  99,  99,  98,  96,  98,  97,  98,  96,
         96,  98,  98, 102, 103, 102, 103, 103, 106, 109,
        112, 114, 113, 114, 117, 118, 121, 125, 130, 138,
        148, 159, 168, 177, 182, 183, 182, 175, 160, 140,
        115,  86,  57,  32,  14,   4,   4,   8,  16,  25,
         32,  40,  46,  50,  55,  59,  59,  58,  58,  57,
         58,  60,  60,  58,  56,  55,  56,  59,  57,  55,
         54,  55,  56,  58,  59,  59,  58,  60,  62,  63,
         66,  65,  65,  66,  67,  69,  69,  69,  68,  69,
         72,  74,  75,  72,  73,  74,  77,  80
    );

begin

    -- Instantiate the Unit Under Test (UUT)
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

    -- 100 MHz System Clock Process (10 ns period)
    clk_process: process
    begin
        clk <= '0';
        wait for clk_period / 2;
        clk <= '1';
        wait for clk_period / 2;
    end process;

    -- Stimulus and Verification Process
    stim_proc: process
    begin
        -- ====================================================================
        -- 1. Initialize and Reset System
        -- ====================================================================
        report "==========================================================" severity note;
        report "   STARTING HARDWARE SNN SIMULATION (SPARTAN-7 XC7S15)    " severity note;
        report "==========================================================" severity note;
        
        rst   <= '1';
        start <= '0';
        wr_en <= '0';
        wait for 100 ns;
        rst   <= '0';
        wait for clk_period * 5;

        -- ====================================================================
        -- TEST CASE 1: Normal Sinus Beat (N)
        -- ====================================================================
        report "----------------------------------------------------------" severity note;
        report ">> TEST 1: Loading Normal Sinus Beat (Class 0: N)..." severity note;
        report "----------------------------------------------------------" severity note;

        -- Load 128 samples into FPGA input register file
        for i in 0 to 127 loop
            addr_in <= std_logic_vector(to_unsigned(i, 7));
            data_in <= std_logic_vector(to_unsigned(PATIENT_BEAT_N(i), 8));
            wr_en   <= '1';
            wait for clk_period;
        end loop;
        wr_en <= '0';
        wait for clk_period * 2;

        -- Trigger SNN classification
        report ">> Starting SNN Inference across 25 Timesteps..." severity note;
        start <= '1';
        wait for clk_period;
        start <= '0';

        -- Wait for SNN hardware to finish processing
        wait until done = '1';
        wait for clk_period * 2;

        -- Verify predicted class
        if class_out = "000" then
            report "[PASSED] Test 1: Hardware Output = Class 0 (Normal 'N') - MATCH!" severity note;
        else
            report "[FAILED] Test 1: Expected Class 0 (N), but Hardware gave: " & 
                   integer'image(to_integer(unsigned(class_out))) severity error;
        end if;

        wait for clk_period * 10;

        -- ====================================================================
        -- TEST CASE 2: Premature Ventricular Contraction (V)
        -- ====================================================================
        report "----------------------------------------------------------" severity note;
        report ">> TEST 2: Loading Ventricular Ectopic Beat (Class 2: V)..." severity note;
        report "----------------------------------------------------------" severity note;

        -- Load 128 samples into FPGA input register file
        for i in 0 to 127 loop
            addr_in <= std_logic_vector(to_unsigned(i, 7));
            data_in <= std_logic_vector(to_unsigned(PATIENT_BEAT_V(i), 8));
            wr_en   <= '1';
            wait for clk_period;
        end loop;
        wr_en <= '0';
        wait for clk_period * 2;

        -- Trigger SNN classification
        report ">> Starting SNN Inference across 25 Timesteps..." severity note;
        start <= '1';
        wait for clk_period;
        start <= '0';

        -- Wait for SNN hardware to finish processing
        wait until done = '1';
        wait for clk_period * 2;

        -- Verify predicted class
        if class_out = "010" then
            report "[PASSED] Test 2: Hardware Output = Class 2 (Ventricular 'V') - MATCH!" severity note;
        else
            report "[FAILED] Test 2: Expected Class 2 (V), but Hardware gave: " & 
                   integer'image(to_integer(unsigned(class_out))) severity error;
        end if;

        report "==========================================================" severity note;
        report ">> SIMULATION COMPLETE: ALL TESTS FINISHED SUCCESSFULLY!  " severity note;
        report "==========================================================" severity note;

        wait;
    end process;

end behavior;
