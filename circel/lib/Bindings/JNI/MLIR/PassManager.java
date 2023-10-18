package MLIR;

import MLIR.Operations.Operation;

public class PassManager extends NativeObject {
  public static PassManager create(Builder builder) {
    return create(PassManager.class, builder);
  }

  protected static native PassManager create(Class ManagerClass, Builder builder);

  public native void enableVerifier();

  /// returns - true if the pass pipline was successful
  public native boolean run(Operation operation);

  protected PassManager(long handle) {
    super(handle);
  }

  // -- MLIR Passes

  /**
   * In MLIR Passes are mostly represented as `unique_ptr<Pass>`, and that value is more often than
   * not passed directly to `PassManager::addPass`. Since some bound languages don't have support
   * for something like `unique_ptr`, we simply define all of the `create...` methods directly on
   * PassManager as `add...` methods.
   */

  public native void addPrintIRPass();
}
