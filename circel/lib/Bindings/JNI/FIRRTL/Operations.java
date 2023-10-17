package FIRRTL;

import MLIR.Builder;
import MLIR.Types.*;
import MLIR.Attributes.*;
import MLIR.Values.*;
import MLIR.Locations.Location;
import MLIR.Region;
import FIRRTL.Types.*;

public class Operations extends MLIR.Operations {
	public static class Circuit extends Operation {

		public static native Circuit build(Builder builder, Location location, String name);

		protected Circuit(long reference) {
			super(reference);
		}
	}
}
