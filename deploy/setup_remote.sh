#!/usr/bin/env bash
# Set up a rented Linux box (Ubuntu 22.04+, x86_64 or arm64) to run STS2-bot evaluations
# and self-play. No root needed. Run ON THE REMOTE after deploy/push.sh copied the files:
#   bash ~/sts2/deploy/setup_remote.sh
# The game DLLs (sts2-cli/lib) are the user's own copy of the game: they stay in this
# private directory (chmod 700) and are never published.
set -euo pipefail
ROOT="${STS2_ROOT:-$HOME/sts2}"
cd "$ROOT"
chmod 700 "$ROOT"
test -f sts2-cli/lib/sts2.dll || { echo "missing sts2-cli/lib (run deploy/push.sh on the Mac first)"; exit 1; }

# .NET 9 SDK, user-local
if ! "$HOME/.dotnet/dotnet" --version >/dev/null 2>&1; then
  curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
  bash /tmp/dotnet-install.sh --channel 9.0 --install-dir "$HOME/.dotnet"
fi
export PATH="$HOME/.dotnet:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1

# engine build (Release, same layout as on the Mac)
(cd sts2-cli/src/Sts2Headless && dotnet build -c Release 2>&1 | tail -2)

# Python env (CPU torch)
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q numpy scikit-learn scipy
fi

# smoke test: 2 short runs with the frozen v40 networks
cd bot
OMP_NUM_THREADS=1 ../.venv/bin/python run.py 2 --acts 1 --ascension 10 --seed-prefix full --workers 2 \
  --rollouts '{"k":6,"m":4,"turns":3}' --deck-nets models/frozen/v40.json --tag remote_smoke | tail -3
echo "cores: $(nproc)  mem: $(free -g | awk '/Mem/{print $2}') GB"
