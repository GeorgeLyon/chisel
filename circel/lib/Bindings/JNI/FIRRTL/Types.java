package FIRRTL;

import MLIR.Context;

public class Types extends MLIR.Types {
	public static class Clock extends Type {
		public static native Clock get(Context context);

		protected Clock(long reference) {
			super(reference);
		}
	}

	public static class UInt extends Type {
		public static native UInt get(Context context, long width);

		protected UInt(long reference) {
			super(reference);
		}
	}
}
