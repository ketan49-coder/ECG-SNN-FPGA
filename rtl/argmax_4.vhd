----------------------------------------------------------------------------------
-- Module: 4-Input Argmax Classifier
-- Project: ECG-SNN-FPGA
-- Description:
--   Combinational comparator tree that finds the index (0..3) of the
--   input with the highest value. Used to determine which of the 4
--   output classes (N, S, V, F) has accumulated the most spikes.
--
--   Output encoding:
--     00 = Normal (N)
--     01 = Supraventricular (S)
--     10 = Ventricular (V)
--     11 = Fusion (F)
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity argmax_4 is
    port (
        val_0     : in  signed(15 downto 0);  -- Class N score
        val_1     : in  signed(15 downto 0);  -- Class S score
        val_2     : in  signed(15 downto 0);  -- Class V score
        val_3     : in  signed(15 downto 0);  -- Class F score
        class_out : out std_logic_vector(1 downto 0)
    );
end argmax_4;

architecture rtl of argmax_4 is
begin

    process(val_0, val_1, val_2, val_3)
        variable max_val : signed(15 downto 0);
        variable max_idx : unsigned(1 downto 0);
    begin
        -- Start with class 0 (Normal) as default
        max_val := val_0;
        max_idx := "00";

        if val_1 > max_val then
            max_val := val_1;
            max_idx := "01";
        end if;

        if val_2 > max_val then
            max_val := val_2;
            max_idx := "10";
        end if;

        if val_3 > max_val then
            max_val := val_3;
            max_idx := "11";
        end if;

        class_out <= std_logic_vector(max_idx);
    end process;

end rtl;
