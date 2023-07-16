module {
  firrtl.circuit "GCD" {
    firrtl.module @GCD(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io: !firrtl.bundle<a flip: uint<32>, b flip: uint<32>, loadValues flip: uint<1>, result: uint<32>, resultIsValid: uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
      %0 = firrtl.subfield %io[resultIsValid] : !firrtl.bundle<a flip: uint<32>, b flip: uint<32>, loadValues flip: uint<1>, result: uint<32>, resultIsValid: uint<1>>
      %1 = firrtl.subfield %io[result] : !firrtl.bundle<a flip: uint<32>, b flip: uint<32>, loadValues flip: uint<1>, result: uint<32>, resultIsValid: uint<1>>
      %2 = firrtl.subfield %io[b] : !firrtl.bundle<a flip: uint<32>, b flip: uint<32>, loadValues flip: uint<1>, result: uint<32>, resultIsValid: uint<1>>
      %3 = firrtl.subfield %io[a] : !firrtl.bundle<a flip: uint<32>, b flip: uint<32>, loadValues flip: uint<1>, result: uint<32>, resultIsValid: uint<1>>
      %4 = firrtl.subfield %io[loadValues] : !firrtl.bundle<a flip: uint<32>, b flip: uint<32>, loadValues flip: uint<1>, result: uint<32>, resultIsValid: uint<1>>
      %x = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<32>
      %y = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<32>
      %5 = firrtl.gt %x, %y : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<1>
      %_T = firrtl.node interesting_name %5 : !firrtl.uint<1>
      firrtl.when %_T : !firrtl.uint<1> {
        %7 = firrtl.sub %x, %y : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
        %_x_T = firrtl.node interesting_name %7 : !firrtl.uint<33>
        %8 = firrtl.tail %_x_T, 1 : (!firrtl.uint<33>) -> !firrtl.uint<32>
        %_x_T_1 = firrtl.node interesting_name %8 : !firrtl.uint<32>
        firrtl.strictconnect %x, %_x_T_1 : !firrtl.uint<32>
      } else {
        %7 = firrtl.sub %y, %x : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
        %_y_T = firrtl.node interesting_name %7 : !firrtl.uint<33>
        %8 = firrtl.tail %_y_T, 1 : (!firrtl.uint<33>) -> !firrtl.uint<32>
        %_y_T_1 = firrtl.node interesting_name %8 : !firrtl.uint<32>
        firrtl.strictconnect %y, %_y_T_1 : !firrtl.uint<32>
      }
      firrtl.when %4 : !firrtl.uint<1> {
        firrtl.strictconnect %x, %3 : !firrtl.uint<32>
        firrtl.strictconnect %y, %2 : !firrtl.uint<32>
      }
      firrtl.strictconnect %1, %x : !firrtl.uint<32>
      %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
      %6 = firrtl.eq %y, %c0_ui1 : (!firrtl.uint<32>, !firrtl.const.uint<1>) -> !firrtl.uint<1>
      %_io_resultIsValid_T = firrtl.node interesting_name %6 : !firrtl.uint<1>
      firrtl.strictconnect %0, %_io_resultIsValid_T : !firrtl.uint<1>
    }
  }
}
