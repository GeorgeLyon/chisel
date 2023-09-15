
package MLIR;

import MLIR.Operations.*;
import MLIR.Region;

public class Builder extends NativeObject {
	protected Builder(long reference) {
		super(reference);
	}

	public static native Builder create(Context context);

	public native void createBlock(Region region);

	public native void setInsertionPointToStart(Block block);

	public native void setInsertionPointToEnd(Block block);

	public native void setInsertionPointBefore(Operation operation);

	public native void setInsertionPointAfter(Operation operation);

	public native void verifyAndPrint(Operations.Module module);
}