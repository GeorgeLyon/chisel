package MLIR;

public class Values {

	public static class Value extends NativeObject {
		protected Value(long pointer) {
			super(pointer);
		}

		public native void dump(Builder builder);
	}

	public static class OpResult extends Value {
		protected OpResult(long pointer) {
			super(pointer);
		}
	}
}