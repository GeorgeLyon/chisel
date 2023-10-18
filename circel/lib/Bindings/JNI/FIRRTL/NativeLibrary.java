package FIRRTL;

import java.util.ArrayList;

public class NativeLibrary extends MLIR.NativeLibrary {
  public static void ensureIsLoaded() throws java.io.IOException {
    (new NativeLibrary()).doEnsureIsLoaded();
  }

  protected NativeLibrary() {}

  protected String getPrimaryLibraryName() {
    return "FIRRTLJNI";
  }

  protected String[] getLibraryNames() {
    // Java doesn't seem to have an array concat..
    ArrayList<String> libraryNames = new ArrayList<String>();
    for (String libraryName : super.getLibraryNames()) {
      libraryNames.add(libraryName);
    }
    libraryNames.add("FIRRTLJNI");
    return libraryNames.toArray(new String[] {});
  }
}
