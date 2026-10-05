library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_pkg.ALL;

entity snn_top_tb is
-- Testbench has no external ports
end snn_top_tb;

architecture behavior of snn_top_tb is

    -- Component Declaration for the Unit Under Test (UUT)
    component snn_top
    port(
        clk          : in std_logic;
        rst          : in std_logic;
        start        : in std_logic;
        input_vector : in input_array;
        done         : out std_logic;
        pred_class   : out std_logic_vector(2 downto 0)
    );
    end component;

    -- Inputs
    signal clk          : std_logic := '0';
    signal rst          : std_logic := '0';
    signal start        : std_logic := '0';
    signal input_vector : input_array := (others => (others => '0'));

    -- Outputs
    signal done         : std_logic;
    signal pred_class   : std_logic_vector(2 downto 0);

    -- Clock period definitions (100 MHz)
    constant clk_period : time := 10 ns;

begin

    -- Instantiate the Unit Under Test (UUT)
    uut: snn_top port map (
        clk          => clk,
        rst          => rst,
        start        => start,
        input_vector => input_vector,
        done         => done,
        pred_class   => pred_class
    );

    -- Clock process definitions
    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    -- Stimulus process
    stim_proc: process
    begin		
        -- 1. Hold reset state for 100 ns.
        rst <= '1';
        wait for 100 ns;	
        rst <= '0';
        wait for clk_period*10;

        -- 2. Load a fake "Patient Heartbeat" into the input pins
        -- We'll simulate a strong peak in the middle of the 128-window
        for i in 0 to NUM_INPUTS-1 loop
            if (i > 60 and i < 68) then
                input_vector(i) <= to_unsigned(200, 8); -- High ECG value (Peak)
            else
                input_vector(i) <= to_unsigned(25, 8);  -- Low ECG value (Baseline)
            end if;
        end loop;

        -- 3. Trigger the FPGA to start calculating
        start <= '1';
        wait for clk_period;
        start <= '0';

        -- 4. Wait for the FPGA to finish (Wait for 'done' signal)
        wait until done = '1';
        
        -- 5. Give it a few clock cycles to settle, then end simulation
        wait for clk_period*10;
        
        -- Stop simulation
        assert false report "Simulation Finished successfully!" severity note;
        wait;
    end process;

end behavior;
