package CircelJNI;

import cz.adamh.utils.NativeUtils;

public final class IRBuilder implements java.io.Closeable {
	public IRBuilder() {
		if (!isInitialized) {
			isInitialized = true;
			initialize();
		}
		nativeReference = createNativeReference();
	}

	public void close() {
		if (!isDestroyed) {
			isDestroyed = true;
			destroy();
		}
	}

	private long nativeReference;
	private boolean isDestroyed = false;

	static {
		try {
			NativeUtils.loadLibraryFromJar("/libCircelJNINative.so");
		} catch (java.io.IOException e1) {
			throw new RuntimeException(e1);
		}
	}

	public static void main(String[] args) {
		try (IRBuilder builder = new IRBuilder()) {
			builder.test();
		}
	}

	private static boolean isInitialized = false;

	private static native void initialize();

	private static native long createNativeReference();

	private native void destroy();

	public native void test();

}
