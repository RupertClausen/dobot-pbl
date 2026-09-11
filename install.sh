#!/usr/bin/env bash
# Set this workspace up on a machine.
#
#   ./install.sh              pick automatically: Docker if it is usable, else a venv
#   ./install.sh --native     force the plain virtualenv path (no Docker)
#   ./install.sh --docker     force the container path
#
# Docker is optional. The library in src/ has no container dependencies - the
# container is just a tidy way to isolate the toolchain on a machine that has
# Docker. Everything runs the same either way; only the command prefix differs:
#
#   with Docker     dobotpbl python -m src.identify_plate
#   native          python -m src.identify_plate     (venv activated)

set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(pwd)"
VENV="$PROJECT/.venv"

MODE=auto
case "${1:-}" in
    --native) MODE=native ;;
    --docker) MODE=docker ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    "") ;;
    *) echo "install.sh: unknown option '$1' (try --help)" >&2; exit 1 ;;
esac

docker_usable() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }

if [ "$MODE" = auto ]; then
    if docker_usable; then
        MODE=docker
        echo "Docker is available -> container setup. Use --native to override."
    else
        MODE=native
        echo "No usable Docker -> virtualenv setup."
    fi
fi
echo

# --------------------------------------------------------------------------- #
# device access - needed either way, and the native path has no container
# group_add to paper over a missing group membership
# --------------------------------------------------------------------------- #
check_groups() {
    local warn=0
    if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
        echo "WARNING: you are not in the 'dialout' group, so the Dobot's serial"
        echo "         port will not be readable. Fix with:"
        echo "             sudo usermod -aG dialout \"$USER\""
        echo "         then log out and back in (a new shell is not enough)."
        warn=1
    fi
    if ! ls /dev/video* >/dev/null 2>&1; then
        echo "NOTE: no /dev/video* present - plug the camera in before running."
        warn=1
    elif ! head -c0 /dev/video0 2>/dev/null; then
        echo "WARNING: /dev/video0 is not readable by you. On a normal desktop"
        echo "         login the session grants this automatically; over SSH it"
        echo "         does not. If it persists: sudo usermod -aG video \"$USER\""
        warn=1
    fi
    [ "$warn" -eq 0 ] && echo "device access looks fine (dialout group, camera readable)"
    echo
}

# --------------------------------------------------------------------------- #
case "$MODE" in
docker)
    docker_usable || {
        echo "install.sh: --docker was requested but the daemon is not reachable." >&2
        echo "            is it running, and are you in the 'docker' group?" >&2
        exit 1; }
    check_groups
    mkdir -p "$HOME/.local/bin"
    install -m 755 dobotpbl "$HOME/.local/bin/dobotpbl"
    echo "installed dobotpbl -> $HOME/.local/bin/dobotpbl"

    if [ "$PROJECT" != "$HOME/Software/Coding/dobot-lab" ]; then
        echo
        echo "This clone is not at the launcher's default location. Add to ~/.bashrc:"
        echo "    export DOBOTPBL_PROJECT=\"$PROJECT\""
    fi

    echo
    echo "building the image (first time takes a few minutes) ..."
    docker build -t dobot-lab:latest \
        --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" "$PROJECT"
    echo
    echo "done. Check it:  dobotpbl --test"
    ;;

native)
    command -v python3 >/dev/null 2>&1 || {
        echo "install.sh: python3 not found." >&2; exit 1; }

    # 3.10 is the floor: the code annotates with `X | None`. Every module carries
    # `from __future__ import annotations` so those are never evaluated at
    # runtime, but dataclass field types on older versions still bite.
    python3 - <<'PY' || exit 1
import sys
if sys.version_info < (3, 10):
    sys.exit(f"install.sh: need Python 3.10+, found {sys.version.split()[0]}")
PY
    echo "python: $(python3 --version)"
    check_groups

    if [ ! -d "$VENV" ]; then
        echo "creating virtualenv at .venv ..."
        python3 -m venv "$VENV"
    else
        echo "reusing existing .venv"
    fi

    echo "installing dependencies (a few minutes the first time) ..."
    "$VENV/bin/pip" install --upgrade pip >/dev/null
    "$VENV/bin/pip" install -r requirements.txt

    echo
    echo "verifying ..."
    "$VENV/bin/python" -m pytest tests -q

    cat <<EOF

done. To use it:

    cd "$PROJECT"
    source .venv/bin/activate
    python -m src.camera            # list cameras
    python -m src.identify_plate    # etc.

Run everything from the project root so 'src' is importable. The README writes
commands as 'dobotpbl python -m src.X'; without Docker just drop the prefix.

Point scripts at the overhead camera by name if the default is wrong:

    CAMERA_INDEX=C270 python -m src.identify_plate
EOF
    ;;
esac
