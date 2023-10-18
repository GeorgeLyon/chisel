
package MLIR;

import MLIR.Values.*;
import MLIR.Locations.Location;

public class Operations {
  public static class Operation extends NativeObject {
    public native void dump(Builder scope);

    public native int getNumberOfResults(Builder scope);

    public native OpResult getResult(Builder scope, int index);

    protected Operation(long reference) {
      super(reference);
    }
  }

  public static class Module extends Operation {
    public static native Module build(Builder builder, Location location);

    public native Block getBody(Builder builder);

    protected Module(long reference) {
      super(reference);
    }
  }
}
