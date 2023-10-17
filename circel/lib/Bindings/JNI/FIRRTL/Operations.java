package FIRRTL;

import MLIR.*;
import MLIR.Types.*;
import MLIR.Attributes.*;
import MLIR.Values.*;
import MLIR.Locations.Location;
import FIRRTL.Types.*;
import FIRRTL.Attributes.*;

public class Operations extends MLIR.Operations {
	public static class Circuit extends Operation {
		public static native Circuit build(Builder builder, Location location, String name);

		public native Block getBody(Builder builder);

		protected Circuit(long reference) {
			super(reference);
		}
	}

	public static class Module extends Operation {
		public static native Module build(Builder builder, Location location, String name, Convention convention);

		protected Module(long reference) {
			super(reference);
		}
	}
}
