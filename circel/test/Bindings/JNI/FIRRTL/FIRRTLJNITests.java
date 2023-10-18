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

				try (Builder builder = Builder.create(context)) {
					MLIR.Operations.Module mlirModule = MLIR.Operations.Module.build(builder, loc);
					builder.setInsertionPointToStart(mlirModule.getBody(builder));

					Circuit circuitOp = Circuit.build(builder, loc, "test");
					builder.setInsertionPointToStart(circuitOp.getBody(builder));

					FIRRTL.Operations.Module module = FIRRTL.Operations.Module.build(builder, loc, "test", scalarized);

					module.addPort(builder, "clock", clockType, FIRRTL.Operations.Module.PortDirection.In);

					/*-
					// CHECK-NEXT: // -----// IR Dump //----- //
					// CHECK-NEXT: module {
					// CHECK-NEXT:	 firrtl.circuit "test" {
					// CHECK-NEXT:     firrtl.module @test(in %clock: !firrtl.clock) attributes {convention = #firrtl<convention scalarized>} {
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
