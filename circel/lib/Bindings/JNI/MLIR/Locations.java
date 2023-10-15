package MLIR;

public class Locations {
	public static class Location extends NativeObject {
		public native void dump(Context context);

		protected Location(long reference) {
			super(reference);
		}
	}

	public static class Unknown extends Location {
		public static native Unknown get(Context context);

		protected Unknown(long reference) {
			super(reference);
		}
	}

	public static class FileLineColumn extends Location {
		public static native FileLineColumn get(Context context, String filename, int line, int column);

		protected FileLineColumn(long reference) {
			super(reference);
		}
	}

	public static class CallSite extends Location {
		public static native CallSite get(Context context, Location callee, Location caller);

		protected CallSite(long reference) {
			super(reference);
		}
	}

	public static class Name extends Location {
		public static native CallSite get(Context context, String name, Location child);

		protected Name(long reference) {
			super(reference);
		}
	}

	public static class Fused extends Location {
		public static native Fused get(Context context, Location[] locations);

		protected Fused(long reference) {
			super(reference);
		}
	}

}