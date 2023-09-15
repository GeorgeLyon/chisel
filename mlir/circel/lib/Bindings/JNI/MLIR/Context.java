package MLIR;

public class Context extends NativeObject {
	private Context(long reference) {
		super(reference);
	}

	public static native Context create();
}
