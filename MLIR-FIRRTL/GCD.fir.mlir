module {
  firrtl.circuit "GCD" {
    firrtl.module @GCD(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_a: !firrtl.uint<32>, in %io_b: !firrtl.uint<32>, in %io_loadValues: !firrtl.uint<1>, out %io_result: !firrtl.uint<32>, out %io_resultIsValid: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
      %x = firrtl.reg %clock {firrtl.random_init_start = 0 : ui64} : !firrtl.clock, !firrtl.uint<32>
      %y = firrtl.reg %clock {firrtl.random_init_start = 32 : ui64} : !firrtl.clock, !firrtl.uint<32>
      %0 = firrtl.gt %x, %y : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<1>
      %_x_T = firrtl.sub %x, %y {name = "_x_T"} : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %_x_T_1 = firrtl.bits %_x_T 31 to 0 {name = "_x_T_1"} : (!firrtl.uint<33>) -> !firrtl.uint<32>
      %1 = firrtl.mux(%0, %_x_T_1, %x) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
      %_y_T = firrtl.sub %y, %x {name = "_y_T"} : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %_y_T_1 = firrtl.bits %_y_T 31 to 0 {name = "_y_T_1"} : (!firrtl.uint<33>) -> !firrtl.uint<32>
      %2 = firrtl.mux(%0, %y, %_y_T_1) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
      %3 = firrtl.mux(%io_loadValues, %io_a, %1) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
      firrtl.strictconnect %x, %3 : !firrtl.uint<32>
      %4 = firrtl.mux(%io_loadValues, %io_b, %2) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
      firrtl.strictconnect %y, %4 : !firrtl.uint<32>
      firrtl.strictconnect %io_result, %x : !firrtl.uint<32>
      %5 = firrtl.orr %y : (!firrtl.uint<32>) -> !firrtl.uint<1>
      %_io_resultIsValid_T = firrtl.not %5 {name = "_io_resultIsValid_T"} : (!firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.strictconnect %io_resultIsValid, %_io_resultIsValid_T : !firrtl.uint<1>
    }
  }
}
