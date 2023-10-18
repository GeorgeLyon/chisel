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

		public enum PortDirection {
			In, Out
		}

		public BlockArgument addPort(Builder builder, String name, Type type, PortDirection direction) {
			switch (direction) {
			case In:
				return addPort(builder, name, type, true);
			case Out:
				return addPort(builder, name, type, false);
			default:
				throw new IllegalArgumentException("Invalid direction: " + direction);
			}
		}

		private native BlockArgument addPort(Builder builder, String name, Type type, boolean directionIsIn);

		protected Module(long reference) {
			super(reference);
		}
	}
}
