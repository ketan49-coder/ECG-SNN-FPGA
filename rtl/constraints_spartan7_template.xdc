## ==============================================================================
## Xilinx Spartan-7 (XC7S15) Constraints Template
## Module: snn_uart_top
## ==============================================================================
## Tomorrow, once the exact board model is confirmed (QMTech / Seeed Studio / etc.),
## update the PACKAGE_PIN values below matching your board's user manual silkscreen.
## ==============================================================================

## 1. System Clock (Default 100 MHz, 10.0 ns period)
# set_property -dict { PACKAGE_PIN <PIN_CLK> IOSTANDARD LVCMOS33 } [get_ports clk]
# create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk]

## 2. Active-Low Pushbutton Reset
# set_property -dict { PACKAGE_PIN <PIN_RST> IOSTANDARD LVCMOS33 } [get_ports rst_n]

## 3. USB-UART Interface (115200 Baud)
# set_property -dict { PACKAGE_PIN <PIN_RX>  IOSTANDARD LVCMOS33 } [get_ports uart_rxd]
# set_property -dict { PACKAGE_PIN <PIN_TX>  IOSTANDARD LVCMOS33 } [get_ports uart_txd]

## 4. 4 On-Board LEDs (Class Indicators: 0=N, 1=S, 2=V, 3=F)
# set_property -dict { PACKAGE_PIN <PIN_LED0> IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
# set_property -dict { PACKAGE_PIN <PIN_LED1> IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
# set_property -dict { PACKAGE_PIN <PIN_LED2> IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
# set_property -dict { PACKAGE_PIN <PIN_LED3> IOSTANDARD LVCMOS33 } [get_ports {led[3]}]
