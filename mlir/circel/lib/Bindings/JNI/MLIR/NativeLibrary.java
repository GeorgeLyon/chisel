package MLIR;

import java.io.*;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.ProviderNotFoundException;
import java.nio.file.StandardCopyOption;

public class NativeLibrary {
	public static void load() throws IOException {
		(new NativeLibrary()).doLoad();
	}

	protected NativeLibrary() {
	}

	protected String getPrimaryLibraryName() {
		return "MLIRJNI";
	}

	protected String[] getLibraryNames() {
		return new String[] { "MLIRJNI" };
	}

	private static boolean isLoaded = false;

	private static synchronized void willLoad() {
		assert (!isLoaded);
		isLoaded = true;
	}

	public void doLoad() throws IOException {
		NativeLibrary.willLoad();

		String libPath = System.getenv("MLIRJNI_LIB_PATH");
		if (libPath != null) {
			/**
			 * MLIRJNI_LIB_PATH is used to override the default behavior of loading the
			 * static libraries from the jar, instead selecting a library that exists
			 * elsewhere on the system. In this case, we expect the Java library path and
			 * rpath to properly resolve any dependent shared libraries. This is currently
			 * used in the CMake java tests.
			 */
			System.load(libPath);
			return;
		}

		// Needed so we can delete the library after loading it
		assert (isPosixCompliant());

		File temporaryDirectory = createTemporaryDirectory();
		try {
			String primaryLibraryName = getPrimaryLibraryName();
			String[] libraryNames = getLibraryNames();
			File primaryLibrary = null;
			for (String libraryName : libraryNames) {
				String libraryFilename = "lib" + libraryName + ".jni";
				File temporaryLibrary = new File(temporaryDirectory, libraryFilename);
				try (InputStream stream = NativeLibrary.class.getResourceAsStream("/" + libraryFilename)) {
					assert (stream != null);
					Files.copy(stream, temporaryLibrary.toPath());
				}
				if (libraryName == primaryLibraryName) {
					primaryLibrary = temporaryLibrary;
				}
			}

			assert (primaryLibrary != null);
			System.load(primaryLibrary.getAbsolutePath());

		} finally {
			temporaryDirectory.delete();
		}

	}

	private static File createTemporaryDirectory() throws IOException {
		String tempDir = System.getProperty("java.io.tmpdir");
		File generatedDir = new File(tempDir, "MLIRJNI_" + System.nanoTime());

		if (!generatedDir.mkdir())
			throw new IOException("Failed to create temp directory " + generatedDir.getName());

		return generatedDir;
	}

	private static boolean isPosixCompliant() {
		try {
			return FileSystems.getDefault().supportedFileAttributeViews().contains("posix");
		} catch (FileSystemNotFoundException | ProviderNotFoundException | SecurityException e) {
			return false;
		}
	}
}
