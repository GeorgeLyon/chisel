
package MLIR;

public class Attributes {
	public static class Attribute extends NativeObject {
		protected Attribute(long reference) {
			super(reference);
		}

		public native void dump(Context context);
	}

	public static class StringAttribute extends Attribute {
		protected StringAttribute(long reference) {
			super(reference);
		}

		public static native StringAttribute get(Context context, String value);
	}
}