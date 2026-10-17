#!/bin/zsh
# winsync.sh HOST [ROOT]: copy the working tree's tracked code (uncommitted edits included)
# to a Windows training machine over SSH. Game DLLs (sts2-cli/lib) and models/*.pt stay put.
set -e
host=$1; root=${2:-D:/sts2}
cd "$(dirname "$0")/.."
git ls-files -co --exclude-standard bot deploy harness sts2-cli/src sts2-cli/localization_eng sts2-cli/localization_zhs \
    | grep -v '\.pt$' | COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf - -T - | ssh -o BatchMode=yes "$host" "tar -xf - -C $root"
echo "synced to $host:$root"
