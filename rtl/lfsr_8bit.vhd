----------------------------------------------------------------------------------
-- Module: 8-bit Linear Feedback Shift Register (LFSR)
-- Project: ECG-SNN-FPGA
-- Description:
--   Generates pseudo-random 8-bit numbers for the Rate Encoder.
--   Uses polynomial: x^8 + x^6 + x^5 + x^4 + 1 (maximal-length, period=255)
--   The LFSR output is compared against normalized input amplitudes to
--   probabilistically generate spikes — this is the core of rate coding.
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity lfsr_8bit is
    port (
        clk      : in  std_logic;
        rst      : in  std_logic;
        en       : in  std_logic;
        lfsr_out : out std_logic_vector(7 downto 0)
    );
end lfsr_8bit;

architecture rtl of lfsr_8bit is
    -- Non-zero seed (LFSR must never be all-zeros or it locks up)
    signal lfsr_reg : std_logic_vector(7 downto 0) := x"A5";
begin

    process(clk)
        variable feedback : std_logic;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                lfsr_reg <= x"A5";  -- Reset to seed
            elsif en = '1' then
                -- Taps at positions 8, 6, 5, 4 (1-indexed)
                -- For 0-indexed: bits 7, 5, 4, 3
                feedback := lfsr_reg(7) xor lfsr_reg(5) xor lfsr_reg(4) xor lfsr_reg(3);
                lfsr_reg <= lfsr_reg(6 downto 0) & feedback;
            end if;
        end if;
    end process;

    lfsr_out <= lfsr_reg;

end rtl;
