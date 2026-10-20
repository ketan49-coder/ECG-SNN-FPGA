----------------------------------------------------------------------------------
-- Module: SNN UART Top-Level (snn_uart_top)
-- Project: ECG-SNN-FPGA
-- Description:
--   Hardware bridge connecting PC/Laptop USB serial port to the 4-class
--   multiplier-free SNN core.
--
-- Protocol:
--   1. PC sends Sync Byte: 0xA5
--   2. PC streams 128 ECG sample bytes (0 to 127)
--   3. FPGA pulses snn_start and executes SNN (4 ms @ 100 MHz)
--   4. Upon completion, FPGA sends 2 response bytes back to PC:
--        Byte 0: 0x5A (Response Header / ACK)
--        Byte 1: 0x00 (N), 0x01 (S), 0x02 (V), or 0x03 (F)
--   5. 4 Physical Board LEDs light up to display diagnosis directly on hardware!
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_pkg.ALL;

entity snn_uart_top is
    generic (
        CLK_FREQ_HZ : integer := 100_000_000;
        BAUD_RATE   : integer := 115_200
    );
    port (
        clk      : in  std_logic;                      -- Board clock oscillator
        rst_n    : in  std_logic;                      -- Active-low pushbutton reset
        uart_rxd : in  std_logic;                      -- USB-UART RX line (from PC)
        uart_txd : out std_logic;                      -- USB-UART TX line (to PC)
        led      : out std_logic_vector(3 downto 0)    -- 4 Board LEDs (Class Indicators)
    );
end snn_uart_top;

architecture rtl of snn_uart_top is

    constant CLKS_PER_BIT : integer := CLK_FREQ_HZ / BAUD_RATE;
    signal rst : std_logic;

    -- UART RX signals
    signal rx_data  : std_logic_vector(7 downto 0);
    signal rx_valid : std_logic;

    -- UART TX signals
    signal tx_data  : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_start : std_logic := '0';
    signal tx_busy  : std_logic;
    signal tx_done  : std_logic;

    -- SNN Core signals
    signal snn_data_in   : std_logic_vector(7 downto 0) := (others => '0');
    signal snn_addr_in   : std_logic_vector(6 downto 0) := (others => '0');
    signal snn_wr_en     : std_logic := '0';
    signal snn_start     : std_logic := '0';
    signal snn_class_out : std_logic_vector(2 downto 0);
    signal snn_done      : std_logic;
    signal snn_busy      : std_logic;

    -- FSM States
    type state_type is (WAIT_SYNC, RECEIVE_BEAT, RUN_SNN, SEND_ACK, SEND_CLASS, DONE_CYCLE);
    signal state : state_type := WAIT_SYNC;

    signal sample_counter : integer range 0 to 128 := 0;
    signal captured_class : std_logic_vector(2 downto 0) := (others => '0');
    signal led_reg        : std_logic_vector(3 downto 0) := "0001"; -- Default LED0 on at boot

    -- Activity heartbeat counter for visual confirmation
    signal heartbeat_cnt  : unsigned(25 downto 0) := (others => '0');

begin

    -- Active-low to active-high reset
    rst <= not rst_n;
    led <= led_reg;

    --------------------------------------------------------------------------------
    -- 1. UART Receiver (Listens to PC USB)
    --------------------------------------------------------------------------------
    u_uart_rx: entity work.uart_rx
        generic map (
            CLKS_PER_BIT => CLKS_PER_BIT
        )
        port map (
            clk      => clk,
            rst      => rst,
            rx       => uart_rxd,
            rx_data  => rx_data,
            rx_valid => rx_valid
        );

    --------------------------------------------------------------------------------
    -- 2. UART Transmitter (Replies to PC USB)
    --------------------------------------------------------------------------------
    u_uart_tx: entity work.uart_tx
        generic map (
            CLKS_PER_BIT => CLKS_PER_BIT
        )
        port map (
            clk      => clk,
            rst      => rst,
            tx_data  => tx_data,
            tx_start => tx_start,
            tx       => uart_txd,
            tx_busy  => tx_busy,
            tx_done  => tx_done
        );

    --------------------------------------------------------------------------------
    -- 3. SNN Neural Network Core
    --------------------------------------------------------------------------------
    u_snn_core: entity work.snn_top
        port map (
            clk       => clk,
            rst       => rst,
            data_in   => snn_data_in,
            addr_in   => snn_addr_in,
            wr_en     => snn_wr_en,
            start     => snn_start,
            class_out => snn_class_out,
            done      => snn_done,
            busy      => snn_busy
        );

    --------------------------------------------------------------------------------
    -- 4. Main Control FSM
    --------------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state          <= WAIT_SYNC;
                sample_counter <= 0;
                snn_wr_en      <= '0';
                snn_start      <= '0';
                tx_start       <= '0';
                tx_data        <= (others => '0');
                captured_class <= (others => '0');
                led_reg        <= "0001";
                heartbeat_cnt  <= (others => '0');
            else
                heartbeat_cnt <= heartbeat_cnt + 1;
                snn_wr_en     <= '0';
                snn_start     <= '0';
                tx_start      <= '0';

                case state is
                    -- Wait for 0xA5 synchronization header from laptop
                    when WAIT_SYNC =>
                        sample_counter <= 0;
                        if rx_valid = '1' and rx_data = x"A5" then
                            state <= RECEIVE_BEAT;
                        end if;

                    -- Receive exactly 128 ECG voltage bytes (0..127)
                    when RECEIVE_BEAT =>
                        if rx_valid = '1' then
                            snn_addr_in <= std_logic_vector(to_unsigned(sample_counter, 7));
                            snn_data_in <= rx_data;
                            snn_wr_en   <= '1';

                            if sample_counter = 127 then
                                sample_counter <= 0;
                                snn_start      <= '1';  -- Trigger classification
                                state          <= RUN_SNN;
                            else
                                sample_counter <= sample_counter + 1;
                            end if;
                        end if;

                    -- Wait for SNN inference to complete
                    when RUN_SNN =>
                        -- Blink LEDs rapidly during inference
                        led_reg <= (others => heartbeat_cnt(22));

                        if snn_done = '1' then
                            captured_class <= snn_class_out;

                            -- Light up specific LED corresponding to predicted class:
                            -- LED 0 = Normal (N)
                            -- LED 1 = Supraventricular (S)
                            -- LED 2 = Ventricular Ectopic (V)
                            -- LED 3 = Fusion (F)
                            case snn_class_out is
                                when "000"  => led_reg <= "0001"; -- Normal
                                when "001"  => led_reg <= "0010"; -- S
                                when "010"  => led_reg <= "0100"; -- V (Dangerous)
                                when "011"  => led_reg <= "1000"; -- F
                                when others => led_reg <= "1111";
                            end case;

                            -- Send ACK byte 0x5A
                            tx_data  <= x"5A";
                            tx_start <= '1';
                            state    <= SEND_ACK;
                        end if;

                    -- Wait for ACK transmission to complete
                    when SEND_ACK =>
                        if tx_done = '1' then
                            -- Send predicted class byte
                            tx_data  <= "00000" & captured_class;
                            tx_start <= '1';
                            state    <= SEND_CLASS;
                        end if;

                    -- Wait for Class byte transmission to complete
                    when SEND_CLASS =>
                        if tx_done = '1' then
                            state <= DONE_CYCLE;
                        end if;

                    when DONE_CYCLE =>
                        state <= WAIT_SYNC;

                    when others =>
                        state <= WAIT_SYNC;
                end case;
            end if;
        end if;
    end process;

end rtl;
