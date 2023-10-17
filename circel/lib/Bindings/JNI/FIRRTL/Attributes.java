package FIRRTL;

import MLIR.Context;

public class Attributes extends MLIR.Attributes {
	public static class Convention extends Attribute {
		public static native Convention getScalarized(Context context);

		protected Convention(long reference) {
			super(reference);
		}
	}
}
