package FIRRTL;

public class PassManager extends MLIR.PassManager {
	public static PassManager create(MLIR.Builder builder) {
		return (PassManager) create(PassManager.class, builder);
	}

	protected PassManager(long reference) {
		super(reference);
	}

	public native void addFirtoolPasses(FirtoolOptions firtoolOptions);
}
