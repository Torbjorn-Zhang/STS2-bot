#!/bin/sh
# Build a glibc-2.28 sts2_rs extension on Apollo (cargo-zigbuild; zig as the linker) for hosts
# with an older glibc than the pod's 2.39 (the labs have 2.31). Needs apollo_install.sh's toolchain
# and patched source; cargo runs inside the source tree, whose rust-toolchain file picks the
# toolchain (rustup has no default). Result: $R/py-glibc228/sts2_rs.abi3.so. Log: $R/zigbuild.log
# ("ZIG_OK" / "ZIG_FAIL"). nice 19, 6 threads, build products in the pod-local overlay.
export R=/scratch/xkzhang/slzhang_cvpr27/rs
export ZIGV=0.13.0
export RUSTUP_HOME=$R/rustup CARGO_HOME=$R/cargo
export PATH=$R/zig-linux-x86_64-$ZIGV:$CARGO_HOME/bin:$PATH CARGO_TARGET_DIR=/tmp/slz-rs-target-zig
LOG=$R/zigbuild.log
# the steps run in their own `sh -e` (set -e is void inside a list tested by || / &&)
sh -e <<'EOS' >> $LOG 2>&1 || echo "ZIG_FAIL $(date)" >> $LOG
echo "start $(date)"
if [ ! -x "$R/zig-linux-x86_64-$ZIGV/zig" ]; then
    cd $R && curl -sSL -o zig.tar.xz "https://ziglang.org/download/$ZIGV/zig-linux-x86_64-$ZIGV.tar.xz"
    tar xJf zig.tar.xz && rm -f zig.tar.xz
fi
cd $R/src/repo/sts2-rs
[ -x "$CARGO_HOME/bin/cargo-zigbuild" ] || nice -n 19 cargo install --locked cargo-zigbuild -j 6
PYO3_PYTHON=/scratch/xkzhang/slzhang_cvpr27/sts2/.venv/bin/python \
    nice -n 19 cargo zigbuild --release -j 6 -p sts2-py --features pyo3/extension-module \
    --target x86_64-unknown-linux-gnu.2.28
mkdir -p $R/py-glibc228
cp $CARGO_TARGET_DIR/x86_64-unknown-linux-gnu/release/libsts2_rs.so $R/py-glibc228/sts2_rs.abi3.so.new
mv -f $R/py-glibc228/sts2_rs.abi3.so.new $R/py-glibc228/sts2_rs.abi3.so
echo "ZIG_OK $(date)"
EOS
