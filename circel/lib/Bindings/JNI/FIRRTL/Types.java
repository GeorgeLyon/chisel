package FIRRTL;

import MLIR.Context;

public class Types extends MLIR.Types {
	public static class Clock extends Type {
		public static native Clock get(Context context);

		protected Clock(long reference) {
			super(reference);
		}
	}
}
