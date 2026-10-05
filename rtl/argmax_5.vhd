----------------------------------------------------------------------------------
-- Module: 5-Input Argmax Classifier
-- Project: ECG-SNN-FPGA
-- Description:
--   Combinational comparator tree that finds the index (0..4) of the
--   input with the highest value. Used to determine which of the 5
--   output classes (N, S, V, F, Q) has accumulated the most spikes.
--
--   Output encoding:
--     000 = Normal (N)
--     001 = Supraventricular (S)
--     010 = Ventricular (V)
--     011 = Fusion (F)
--     100 = Unknown (Q)
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity argmax_5 is
    port (
        val_0     : in  signed(15 downto 0);  -- Class N score
        val_1     : in  signed(15 downto 0);  -- Class S score
        val_2     : in  signed(15 downto 0);  -- Class V score
        val_3     : in  signed(15 downto 0);  -- Class F score
        val_4     : in  signed(15 downto 0);  -- Class Q score
        class_out : out std_logic_vector(2 downto 0)
    );
end argmax_5;

architecture rtl of argmax_5 is
begin

    process(val_0, val_1, val_2, val_3, val_4)
        variable max_val : signed(15 downto 0);
        variable max_idx : unsigned(2 downto 0);
    begin
        -- Start with class 0 (Normal) as default
        max_val := val_0;
        max_idx := "000";

        if val_1 > max_val then
            max_val := val_1;
            max_idx := "001";
        end if;

        if val_2 > max_val then
            max_val := val_2;
            max_idx := "010";
        end if;

        if val_3 > max_val then
            max_val := val_3;
            max_idx := "011";
        end if;

        if val_4 > max_val then
            max_val := val_4;
            max_idx := "100";
        end if;

        class_out <= std_logic_vector(max_idx);
    end process;

end rtl;
