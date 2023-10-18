// RUN: MLIRJNI_LIB_PATH=%MLIR_JNI_LIB_DIR%/libFIRRTLJNI.jni circel-java-test-launcher --class FIRRTLJNITests\$Test 2>&1 | FileCheck %s

import MLIR.*;
import MLIR.Attributes.*;
import MLIR.Operations.*;
import MLIR.Types.*;
import MLIR.Values.*;
import MLIR.Locations.Location;
import FIRRTL.*;
import FIRRTL.Attributes.*;
import FIRRTL.Operations.*;
import FIRRTL.Types.*;

public class FIRRTLJNITests {
	public static class Test {
		public static void main(String[] args) {
			try {
				FIRRTL.NativeLibrary.ensureIsLoaded();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			try (Context context = Context.create()) {
				FIRRTL.Dialect.load(context);

				// CHECK: loc(unknown)
				Location loc = Locations.Unknown.get(context);
				loc.dump(context);

				// CHECK-NEXT: #firrtl<convention scalarized>
				Convention scalarized = Convention.getScalarized(context);
				scalarized.dump(context);

				// CHECK-NEXT: !firrtl.clock
				Clock clockType = Clock.get(context);
				clockType.dump(context);

				// CHECK-NEXT: !firrtl.uint<1>
				UInt bitType = UInt.get(context, 1);
				bitType.dump(context);

				// CHECK-NEXT: !firrtl.uint<32>
				UInt uintValueType = UInt.get(context, 32);
				uintValueType.dump(context);

				try (Builder builder = Builder.create(context)) {

					/*-
					// CHECK-NEXT: // -----// IR Dump //----- //
					// CHECK-NEXT: module {
					*/
					MLIR.Operations.Module mlirModule = MLIR.Operations.Module.build(builder, loc);
					builder.setInsertionPointToStart(mlirModule.getBody(builder));

					/*-
					// CHECK-NEXT:	 firrtl.circuit "test" {
					*/
					Circuit circuitOp = Circuit.build(builder, loc, "test");
					builder.setInsertionPointToStart(circuitOp.getBody(builder));

					/*-
					// CHECK-NEXT:     firrtl.module @test(
					*/
					FIRRTL.Operations.Module module = FIRRTL.Operations.Module.build(builder, loc, "test", scalarized);

					// CHECK-SAME: in %clock: !firrtl.clock,
					BlockArgument clockPort = module.addPort(builder, "clock", clockType,
							FIRRTL.Operations.Module.PortDirection.In);
					// CHECK-SAME: in %reset: !firrtl.uint<1>,
					BlockArgument resetPort = module.addPort(builder, "reset", bitType,
							FIRRTL.Operations.Module.PortDirection.In);
					// CHECK-SAME: in %a: !firrtl.uint<32>,
					BlockArgument aPort = module.addPort(builder, "a", uintValueType,
							FIRRTL.Operations.Module.PortDirection.In);
					// CHECK-SAME: in %b: !firrtl.uint<32>,
					BlockArgument bPort = module.addPort(builder, "b", uintValueType,
							FIRRTL.Operations.Module.PortDirection.In);
					// CHECK-SAME: in %loadValues: !firrtl.uint<1>,
					BlockArgument loadValuesPort = module.addPort(builder, "loadValues", bitType,
							FIRRTL.Operations.Module.PortDirection.In);
					// CHECK-SAME: out %result: !firrtl.uint<32>,
					BlockArgument resultPort = module.addPort(builder, "result", uintValueType,
							FIRRTL.Operations.Module.PortDirection.Out);
					// CHECK-SAME: out %resultIsValid: !firrtl.uint<1>
					BlockArgument resultIsValidPort = module.addPort(builder, "resultIsValid", bitType,
							FIRRTL.Operations.Module.PortDirection.Out);

					/*-
					// CHECK-SAME: ) attributes {convention = #firrtl<convention scalarized>} {
					*/
					builder.setInsertionPointToStart(module.getBody(builder));

					/*-
					// CHECK-NEXT: %x = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<32>
					// CHECK-NEXT: %y = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<32>
					*/
					NameKind interestingName = NameKind.getInteresting(context);
					Register x = Register.build(builder, loc, uintValueType, clockPort, "x", interestingName, false);
					Register y = Register.build(builder, loc, uintValueType, clockPort, "y", interestingName, false);

					/*-
					// CHECK-NEXT:     }
					// CHECK-NEXT:   }
					// CHECK-NEXT: }
					// CHECK-NEXT: IR is valid!
					*/
					PassManager passManager = PassManager.create(builder);
					passManager.enableVerifier();
					passManager.addPrintIRPass();
					if (passManager.run(mlirModule)) {
						System.out.println("IR is valid!");
					}
				}
			}
		}
	}
}
