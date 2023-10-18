package MLIR;

import java.lang.ref.Cleaner;

abstract class NativeObject implements AutoCloseable {

  protected NativeObject(long pointer) {
    this.pointer = pointer;
    this.cleanable = sharedCleaner.register(this, new Destroyer(pointer));
  }

  private long pointer;

  private Cleaner.Cleanable cleanable;

  public void close() {
    pointer = 0;
    cleanable.clean();
  }

  /**
   * A runnable which destroys a native pointer
   */
  private static class Destroyer implements Runnable {
    public Destroyer(long pointer) {
      this.pointer = pointer;
    }

    public void run() {
      NativeObject.releaseNativeObject(pointer);
    }

    private long pointer;
  }

  private static final Cleaner sharedCleaner = Cleaner.create();

  private static native void releaseNativeObject(long pointer);
}
