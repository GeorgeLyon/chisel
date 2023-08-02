# MLIR, CIRCT and CIRCEL

Chisel uses CIRCT to generate Verilog, which is built on top of MLIR and the LLVM project infrastructure.

## Version Control Strategy

CIRCT is built on top of MLIR, which is part of the LLVM project. The traditional way to depend on LLVM is to use a submodule, and a nested submodule when depending on something like CIRCT, which itself imports LLVM as a submodule. Not only is this a bit complex to manage, but LLVM is gigantic, with the checkout alone taking up 1.7GB of space at the time of writing. 
This has very real implications for developer productivity: 
- Checkouts are slow, which is particularly painful for CI (1.5 minutes vs 6 seconds in GitHub Actions)
- Common git operations, such as `git status` are slow and sometimes called from the CMake configuration flow
- IDE functionality is also impacted by the massive size of the repo
- Sharing small tweaks to LLVM requires pushing _two_ new branches (one to LLVM, and one to CIRCT)
- Developers are require to `git submodule update --recursive` whenever checking out a commit with a different CIRCT version
- It is difficult to see what changed when updating the CIRCt revision
- The repo takes up more space on developer machines

In light of this, we take a somewhat non-traditional approach to managing the CIRCT dependency and commit the sources we use directly to this repository. The logic for maintaining this live in the `update-circt` script located in `mlir/support`. Most often, all a developer would need to do is run `mlir/support/update-circt origin/main` which updates CIRCT to what is currently in `origin/main`. One can also pass a different revision to `update-circt` (for instance `mlir/support/update-circt my-branch`), or pass the `smash` argument which will override any local modifications to the CIRCT source (this is useful when changing the logic for what bits of CIRCT/LLVM are imported into the project). Local modifications are tracked `mlir/support/{circt,llvm}.diff`, which should be empty in almost all scenarios.
