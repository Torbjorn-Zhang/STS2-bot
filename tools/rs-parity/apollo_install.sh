#!/bin/sh
# Install Rust + the patched sts2-rs on the Apollo pods' shared Lustre dir and build the PyO3
# extension (sts2_rs) and run_arena. Idempotent; log in $R/install.log ("BUILD_OK" / "BUILD_FAIL").
# Needs $R/sts2-rs-local.patch next to it: a changed patch (md5 vs $R/src/patch.md5) re-extracts the
# pristine source from the cached tarball and re-applies it. Compiles with 6 threads at nice 19 (the
# pod runs our self-play jobs); build products go to the pod-local overlay, results are copied to Lustre.
export R=/scratch/xkzhang/slzhang_cvpr27/rs
export SHA=52bfee16387685a415abc22255c7b58e62f0f91e
export RUSTUP_HOME=$R/rustup CARGO_HOME=$R/cargo
export PATH=$CARGO_HOME/bin:$PATH CARGO_TARGET_DIR=/tmp/slz-rs-target
LOG=$R/install.log
mkdir -p $R
# the steps run in their own `sh -e` (set -e is void inside a list tested by || / &&)
sh -e <<'EOS' >> $LOG 2>&1 || echo "BUILD_FAIL $(date)" >> $LOG
echo "start $(date)"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o $R/rustup-init.sh
    sh $R/rustup-init.sh -y --no-modify-path --default-toolchain none --profile minimal
fi
PMD5=$(md5sum $R/sts2-rs-local.patch | cut -d' ' -f1)
if [ ! -d "$R/src/repo/sts2-rs" ] || [ "$(cat $R/src/patch.md5 2>/dev/null)" != "$PMD5" ]; then
    mkdir -p $R/src && cd $R/src
    [ -s sm.tgz ] || curl -sSL -o sm.tgz "https://codeload.github.com/tfriedel/spire-mentor/tar.gz/$SHA"
    rm -rf "spire-mentor-$SHA"
    tar xzf sm.tgz --wildcards "spire-mentor-$SHA/sts2-rs/*"
    (cd "spire-mentor-$SHA" && patch -p1 < $R/sts2-rs-local.patch)
    rm -rf repo && mv "spire-mentor-$SHA" repo
    echo "$PMD5" > patch.md5
    echo "patched $PMD5"
fi
cd $R/src/repo/sts2-rs
PYO3_PYTHON=/scratch/xkzhang/slzhang_cvpr27/sts2/.venv/bin/python \
    nice -n 19 cargo build --release -j 6 -p sts2-py --features pyo3/extension-module
nice -n 19 cargo build --release -j 6 -p sts2-cli --bin run_arena
nice -n 19 cargo build --release -j 6 -p sts2-bench --example site_random_bench
mkdir -p $R/py $R/bin
# a new inode, so processes that already loaded the old .so keep theirs
cp $CARGO_TARGET_DIR/release/libsts2_rs.so $R/py/sts2_rs.abi3.so.new && mv -f $R/py/sts2_rs.abi3.so.new $R/py/sts2_rs.abi3.so
cp $CARGO_TARGET_DIR/release/run_arena $CARGO_TARGET_DIR/release/examples/site_random_bench $R/bin/
echo "BUILD_OK $(date)"
EOS
