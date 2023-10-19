/*-
// RUN: MLIRJNI_LIB_PATH=%MLIR_JNI_LIB_DIR%/libMLIRJNI.jni circel-java-test-launcher --class MLIRJNITests\$Test 2>&1 | FileCheck %s
*/

import MLIR.*;
import MLIR.Attributes.*;
import MLIR.Operations.*;
import MLIR.Types.*;
import MLIR.Locations.*;

public class MLIRJNITests {
  public static class Test {
    public static void main(String[] args) {
      try {
        MLIR.NativeLibrary.ensureIsLoaded();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      try (Context context = Context.create()) {
        // CHECK: Hello from MLIR!
        StringAttribute stringAttribute = StringAttribute.get(context, "Hello from MLIR!");
        stringAttribute.dump(context);

        // Validate that `ensureIsLoaded` does not reload the library (which would break
        // things).
        try {
          MLIR.NativeLibrary.ensureIsLoaded();
        } catch (Exception e) {
          throw new RuntimeException(e);
        }

        // CHECK: loc(unknown)
        Location location = Unknown.get(context);
        location.dump(context);

        try (Builder builder = Builder.create(context)) {
          // CHECK: // -----// IR Dump //----- //
          // CHECK-NEXT: module {
          // CHECK-NEXT: }
          // CHECK-NEXT: true
          Operations.Module module = Operations.Module.build(builder, location);
          PassManager passManager = PassManager.create(builder);
          passManager.enableVerifier();
          passManager.addPrintIRPass();
          System.err.println(passManager.run(module));

          // CHECK: i32
          IntegerType i32 = IntegerType.get(context, 32);
          i32.dump(context);

          // CHECK: i64
          IntegerType i64 = IntegerType.get(context, 64);
          i64.dump(context);

          // CHECK: (i32, i64) -> (i64, i32)
          Type[] arguments = {i32, i64};
          Type[] results = {i64, i32};
          FunctionType functionType = FunctionType.get(context, arguments, results);
          functionType.dump(context);
        }
      }
    }
  }
}
