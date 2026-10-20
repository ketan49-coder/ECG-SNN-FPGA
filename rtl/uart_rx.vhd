----------------------------------------------------------------------------------
-- Module: UART Receiver (uart_rx)
-- Description:
--   Standard 8-N-1 UART receiver with midpoint sampling.
--   Receives serial data on 'rx' line and asserts 'rx_valid' for 1 clock cycle
--   when a new byte is available on 'rx_data'.
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity uart_rx is
    generic (
        CLKS_PER_BIT : integer := 868  -- 100 MHz clock / 115200 baud = 868
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;
        rx       : in  std_logic;
        rx_data  : out std_logic_vector(7 downto 0);
        rx_valid : out std_logic
    );
end uart_rx;

architecture rtl of uart_rx is

    type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT, CLEANUP);
    signal state : state_type := IDLE;

    -- Double-register RX to prevent metastability
    signal rx_sync1 : std_logic := '1';
    signal rx_sync2 : std_logic := '1';

    signal clk_count : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_index : integer range 0 to 7 := 0;
    signal rx_byte   : std_logic_vector(7 downto 0) := (others => '0');

begin

    process(clk)
    begin
        if rising_edge(clk) then
            rx_sync1 <= rx;
            rx_sync2 <= rx_sync1;
        end if;
    end process;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state     <= IDLE;
                rx_valid  <= '0';
                rx_data   <= (others => '0');
                clk_count <= 0;
                bit_index <= 0;
                rx_byte   <= (others => '0');
            else
                rx_valid <= '0';

                case state is
                    when IDLE =>
                        clk_count <= 0;
                        bit_index <= 0;
                        -- Detect falling edge (start bit)
                        if rx_sync2 = '0' then
                            state <= START_BIT;
                        end if;

                    when START_BIT =>
                        -- Check if start bit is still low at midpoint
                        if clk_count = (CLKS_PER_BIT / 2) then
                            if rx_sync2 = '0' then
                                clk_count <= 0;
                                state     <= DATA_BITS;
                            else
                                state     <= IDLE;  -- False alarm glitch
                            end if;
                        else
                            clk_count <= clk_count + 1;
                        end if;

                    when DATA_BITS =>
                        if clk_count = CLKS_PER_BIT - 1 then
                            clk_count <= 0;
                            rx_byte(bit_index) <= rx_sync2;
                            if bit_index = 7 then
                                bit_index <= 0;
                                state     <= STOP_BIT;
                            else
                                bit_index <= bit_index + 1;
                            end if;
                        else
                            clk_count <= clk_count + 1;
                        end if;

                    when STOP_BIT =>
                        -- Wait for stop bit
                        if clk_count = CLKS_PER_BIT - 1 then
                            rx_data   <= rx_byte;
                            rx_valid  <= '1';
                            clk_count <= 0;
                            state     <= CLEANUP;
                        else
                            clk_count <= clk_count + 1;
                        end if;

                    when CLEANUP =>
                        state <= IDLE;

                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

end rtl;
