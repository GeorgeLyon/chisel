package chiselTests.simulator

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

import chisel3._
import chisel3.simulator.EphemeralSimulator._

import CircelJNI._
import scala.util.Using

class EphemeralSimulatorSpec extends AnyFunSpec with Matchers {
  describe("EphemeralSimulator") {
    it("runs GCD correctly") {
      simulate(new GCD()) { gcd =>
        Using(new IRBuilder()) { builder =>
          builder.test()
        }
        gcd.io.a.poke(24.U)
        gcd.io.b.poke(36.U)
        gcd.io.loadValues.poke(1.B)
        gcd.clock.step()
        gcd.io.loadValues.poke(0.B)
        gcd.clock.stepUntil(sentinelPort = gcd.io.resultIsValid, sentinelValue = 1, maxCycles = 10)
        gcd.io.resultIsValid.expect(true.B)
        gcd.io.result.expect(12)
      }
    }
  }
}
