package FIRRTL;

import MLIR.Context;

public class Attributes extends MLIR.Attributes {
	public static class Convention extends Attribute {
		public static native Convention getScalarized(Context context);

		protected Convention(long reference) {
			super(reference);
		}
	}

	public static class NameKind extends Attribute {
		public static native NameKind getDroppable(Context context);

		public static native NameKind getInteresting(Context context);

		protected NameKind(long reference) {
			super(reference);
		}
	}
}
