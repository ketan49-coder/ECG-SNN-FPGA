----------------------------------------------------------------------------------
-- Module: UART Transmitter (uart_tx)
-- Description:
--   Standard 8-N-1 UART transmitter.
--   When 'tx_start' is pulsed, serializes 'tx_data' out onto the 'tx' line.
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity uart_tx is
    generic (
        CLKS_PER_BIT : integer := 868  -- 100 MHz clock / 115200 baud = 868
    );
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;
        tx_data  : in  std_logic_vector(7 downto 0);
        tx_start : in  std_logic;
        tx       : out std_logic;
        tx_busy  : out std_logic;
        tx_done  : out std_logic
    );
end uart_tx;

architecture rtl of uart_tx is

    type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT, CLEANUP);
    signal state : state_type := IDLE;

    signal clk_count : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_index : integer range 0 to 7 := 0;
    signal tx_byte   : std_logic_vector(7 downto 0) := (others => '0');

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state     <= IDLE;
                tx        <= '1';  -- Idle line is high
                tx_busy   <= '0';
                tx_done   <= '0';
                clk_count <= 0;
                bit_index <= 0;
                tx_byte   <= (others => '0');
            else
                tx_done <= '0';

                case state is
                    when IDLE =>
                        tx      <= '1';
                        tx_busy <= '0';
                        clk_count <= 0;
                        bit_index <= 0;

                        if tx_start = '1' then
                            tx_byte <= tx_data;
                            tx_busy <= '1';
                            state   <= START_BIT;
                        end if;

                    when START_BIT =>
                        tx <= '0';  -- Start bit is low
                        if clk_count = CLKS_PER_BIT - 1 then
                            clk_count <= 0;
                            state     <= DATA_BITS;
                        else
                            clk_count <= clk_count + 1;
                        end if;

                    when DATA_BITS =>
                        tx <= tx_byte(bit_index);
                        if clk_count = CLKS_PER_BIT - 1 then
                            clk_count <= 0;
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
                        tx <= '1';  -- Stop bit is high
                        if clk_count = CLKS_PER_BIT - 1 then
                            clk_count <= 0;
                            tx_done   <= '1';
                            state     <= CLEANUP;
                        else
                            clk_count <= clk_count + 1;
                        end if;

                    when CLEANUP =>
                        tx_busy <= '0';
                        state   <= IDLE;

                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

end rtl;
