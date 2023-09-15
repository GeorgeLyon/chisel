// RUN: MLIRJNI_LIB_PATH=%MLIR_LIB_DIRECTORY%/libMLIRJNI.jni java -cp %MLIR_JAVA_CLASSPATH_ARGUMENT% MLIRJNITests\$Test 2>&1 | FileCheck %s

import MLIR.*;
import MLIR.Attributes.*;
import MLIR.Operations.*;
import MLIR.Types.*;
import MLIR.Locations.*;

public class MLIRJNITests {
	public static class Test {
		public static void main(String[] args) {
			try {
				MLIR.NativeLibrary.load();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			try (Context context = Context.create()) {
				// CHECK: Hello from MLIR!
				StringAttribute stringAttribute = StringAttribute.get(context, "Hello from MLIR!");
				stringAttribute.dump(context);

				// CHECK: loc(unknown)
				Location location = Unknown.get(context);
				location.dump(context);

				try (Builder builder = Builder.create(context)) {
					// CHECK: module {
					// CHECK: }
					Operations.Module module = Operations.Module.build(builder, location);
					module.dump(builder);

					// CHECK: i32
					IntegerType i32 = IntegerType.get(context, 32);
					i32.dump(context);

					// CHECK: i64
					IntegerType i64 = IntegerType.get(context, 64);
					i64.dump(context);

					// CHECK: (i32, i64) -> (i64, i32)
					Type[] arguments = { i32, i64 };
					Type[] results = { i64, i32 };
					FunctionType functionType = FunctionType.get(context, arguments, results);
					functionType.dump(context);
				}
			}
		}
	}
}
