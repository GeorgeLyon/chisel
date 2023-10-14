FROM --platform=$BUILDPLATFORM ubuntu:rolling

# The following line is needed to set the $TARGETARCH variable to amd64 or arm64 depending on the target architecture. If you need alternative formulations (x86_64 or aarch64) use $(arch) in a RUN command.
ARG TARGETARCH

# This can be changed to "ci" to remove developer tools and reduce image size. Currently this only saves about 200MB of space.
ARG CONTAINER_MODE=dev

# Different versions of clang and Java can be selected by specifying the following build arguments, but are not guaranteed to work.
ARG CLANG_VERSION=17
ARG JAVA_VERSION=java11

# Can be set to something like `firtool-1.56.1` to use a specific CIRCT release
ARG PINNED_CIRCT_VERSION=

# Setup container environment
ENV \
    # Change locale to suppress some warnings
    LC_ALL=C \
    #
    # Setup clang
    CC=clang-$CLANG_VERSION \
    CXX=clang++-$CLANG_VERSION \
    LLVM_SYMBOLIZER_PATH=llvm-symbolizer-$CLANG_VERSION \
    #
    # Setup Java (GraalVM)
    JAVA_HOME=/graalvm \
    LD_LIBRARY_PATH=/graalvm/lib/server:$LD_LIBRARY_PATH \
    PATH=/graalvm/bin/:$PATH \
    #
    # Disable Chisel argument extensions (which break interactive testing via bloop)
    CHISEL_ARGUMENT_EXTENSIONS=DISABLE \
    #
    # Set VSCode as default editor
    EDITOR="code --wait" \
    GIT_EDITOR="code --wait"

# Setup container
RUN \
    #
    # Disable apt interactive prompts for this RUN command
    export DEBIAN_FRONTEND="noninteractive" && \
    #
    # Get release information (used to determine the correct URL for the LLVM apt repository)
    . /etc/lsb-release && \
    #
    # Get package list
    apt-get update && \
    #
    # Add LLVM/Clang repository so we can install newer versions of clang
    apt-get install -y software-properties-common gnupg curl && \
    curl -L https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/$DISTRIB_CODENAME/ llvm-toolchain-$DISTRIB_CODENAME-$CLANG_VERSION main" && \
    # 
    # Update again now that we've added the LLVM repository
    apt-get update && \
    # 
    # Install required packages
    apt-get install -y \
      #
      # The following packages are used to build CIRCEL
      ccache \
      cmake \
      make \
      ninja-build \
      clang-$CLANG_VERSION \
      clang-tools-$CLANG_VERSION \
      clang-format-$CLANG_VERSION \
      clangd-$CLANG_VERSION \
      lld-$CLANG_VERSION \
      lldb-$CLANG_VERSION \
      jq \
      # 
      # Swift dependencies
      binutils \
      git \
      gnupg2 \
      libc6-dev \
      libcurl4-openssl-dev \
      libedit2 \
      libgcc-9-dev \
      libpython3.8 \
      libsqlite3-0 \
      libstdc++-9-dev \
      libxml2-dev \
      libz3-dev \
      pkg-config \
      tzdata \
      unzip \
      zlib1g-dev \
      #
      # ShellCheck is useful for validating bash scripts
      shellcheck \
      #
      # Verilator is used to run generated verilog
      verilator \
    && \
    #
    # Install Java (GraalVM)
    mkdir /graalvm && \
    curl -sL https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-$JAVA_VERSION-linux-$(arch | sed s/x86_64/amd64/)-22.3.1.tar.gz | tar -xz -C /graalvm --strip-components=1 && \
    #
    # Install Scala (SBT)
    curl -sL https://github.com/sbt/sbt/releases/download/v1.9.6/sbt-1.9.6.tgz | tar xz --strip-components=1 -C /usr && \
    #
    # Install Swift
    mkdir /swift && \
    case $TARGETARCH in \
      "amd64") \
        export SWIFT_URL=https://download.swift.org/swift-5.9-release/ubuntu2204/swift-5.9-RELEASE/swift-5.9-RELEASE-ubuntu22.04.tar.gz \
        ;; \
      "arm64") \
        export SWIFT_URL=https://download.swift.org/swift-5.9-release/ubuntu2204-aarch64/swift-5.9-RELEASE/swift-5.9-RELEASE-ubuntu22.04-aarch64.tar.gz \
        ;; \
      *) \
        echo "Unsupported TARGETARCH: $TARGETARCH" && \
        exit 1 \
        ;; \
    esac && \
    curl -sL "$SWIFT_URL" | tar xz --strip-components=1 -C /swift && \
    ln -s /swift/usr/bin/swift /usr/bin/swift && \
    #
    # Grab CIRCT Sources from GitHub to avoid cloning all of LLVM
    export PINNED_URL_FRAGMENT=${PINNED_CIRCT_VERSION:+download/$PINNED_CIRCT_VERSION} && \
    export URL_FRAGMENT=${PINNED_URL_FRAGMENT:-latest/download} && \
    mkdir /circt && \
    curl -L https://github.com/llvm/circt/releases/$URL_FRAGMENT/circt-full-sources.tar.gz | tar xz --strip-components=1 -C /circt && \
    mv /circt/llvm /llvm && \
    #
    # Remove unnecessary LLVM components
    find /llvm -mindepth 1 -maxdepth 1 ! \( -name llvm -o -name mlir -o -name third-party -o -name cmake \) -print0 | xargs -0 -I {} rm -rf {} && \
    find /llvm/llvm/lib/Target -mindepth 1 -maxdepth 1 -type d ! \( -name AArch64 -o -name X86 -o -name RISCV \) -print0 | xargs -0 -I {} rm -rf {} && \
    rm -rf /llvm/llvm/test/* && touch /llvm/llvm/test/CMakeLists.txt && \
    #
    # Remove GCC (installed as a dependency of CMake) and the MSVC version of clang to reduce CMake confusion
    apt remove -y gcc && \
    apt autoremove -y && \
    rm /usr/bin/clang-cl-$CLANG_VERSION && \
    #
    # Verilator hardcodes "g++" as the C++ compiler, so we symlink clang++ to g++ to avoid installing gcc
    cd /usr/bin && ln -s clang++-$CLANG_VERSION g++ && \
    #
    # Based on the container mode, either remove unnecessary packages or install additional developer-oriented packages
    case "$CONTAINER_MODE" in \
      "ci") \
        #
        # Remove packages we only needed for this step
        apt-add-repository -r "deb http://apt.llvm.org/$UBUNTU_VERSION_NAME/ llvm-toolchain-$UBUNTU_VERSION_NAME-$CLANG_VERSION main" && \
        apt remove -y software-properties-common gnupg curl && \
        #
        # Clean up apt to reduce image size
        apt autoremove -y && \
        rm -rf /var/lib/apt/lists/* \
        ;; \
      "dev") \
        apt-get install -y git \
        ;; \
      *) \
        echo "Invalid CONTAINER_MODE: $CONTAINER_MODE" && \
        exit 1 \
        ;; \
    esac
