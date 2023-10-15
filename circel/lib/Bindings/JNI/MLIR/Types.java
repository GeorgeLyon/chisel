package MLIR;

public class Types {
	public static class Type extends NativeObject {
		protected Type(long reference) {
			super(reference);
		}

		public native void dump(Context context);
	}

	public static class IntegerType extends Type {
		protected IntegerType(long reference) {
			super(reference);
		}

		public native static IntegerType get(Context context, int width);
	}

	public static class FunctionType extends Type {
		protected FunctionType(long reference) {
			super(reference);
		}

		public native static FunctionType get(Context context, Type[] arguments, Type[] results);
	}
}
