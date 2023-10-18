
package MLIR;

import MLIR.Types.*;

public class Attributes {
  public static class Attribute extends NativeObject {
    protected Attribute(long reference) {
      super(reference);
    }

    public native void dump(Context context);
  }

  public static class IntegerAttribute extends Attribute {
    protected IntegerAttribute(long reference) {
      super(reference);
    }

    public static native IntegerAttribute get(Context context, Type type, long value);
  }

  public static class StringAttribute extends Attribute {
    protected StringAttribute(long reference) {
      super(reference);
    }

    public static native StringAttribute get(Context context, String value);
  }

  public static class ArrayAttribute extends Attribute {
    protected ArrayAttribute(long reference) {
      super(reference);
    }

    public static native ArrayAttribute get(Context context, Attribute[] members);
  }
}
