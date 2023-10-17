// RUN: MLIRJNI_LIB_PATH=%MLIR_JNI_LIB_DIR%/libFIRRTLJNI.jni circel-java-test-launcher --class FIRRTLJNITests\$Test 2>&1 | FileCheck %s

import MLIR.*;
import MLIR.Attributes.*;
import MLIR.Operations.*;
import MLIR.Types.*;
import MLIR.Values.*;
import MLIR.Locations.*;
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

				// CHECK: Hello, FIRRTL!
				System.err.println("Hello, FIRRTL!");
			}
		}
	}
}
