// RUN: MLIRJNI_LIB_PATH=%MLIR_JNI_LIB_DIR%/libFIRRTLJNI.jni circel-java-test-launcher --class FIRRTLJNITests\$Test 2>&1 | FileCheck %s

import MLIR.*;
import MLIR.Attributes.*;
import MLIR.Operations.*;
import MLIR.Types.*;
import MLIR.Values.*;
import MLIR.Locations.Location;
import FIRRTL.*;
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

				try (Builder builder = Builder.create(context)) {
					MLIR.Operations.Module moduleOp = MLIR.Operations.Module.build(builder, loc);
					builder.setInsertionPointToStart(moduleOp.getBody(builder));

					/*-
					// CHECK:      module {
					// CHECK-NEXT: }
					// CHECK-NEXT: Run succeeded.
					*/
					PassManager passManager = PassManager.create(builder);
					passManager.enableVerifier();
					passManager.addPrintIRPass();
					if (passManager.run(moduleOp)) {
						System.out.println("Run succeeded.");
					}
				}
			}
		}
	}
}
