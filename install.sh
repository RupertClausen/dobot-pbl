#!/usr/bin/env bash
# Set this workspace up on a new machine (e.g. the lab computer).
#
#   git clone git@github.com:RupertClausen/dobot-pbl.git
#   cd dobot-pbl && ./install.sh
#
# Installs the `dobotpbl` launcher into ~/.local/bin and builds the image.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(pwd)"

command -v docker >/dev/null 2>&1 || {
    echo "install.sh: docker is not installed." >&2; exit 1; }
docker info >/dev/null 2>&1 || {
    echo "install.sh: cannot talk to the docker daemon." >&2
    echo "            is it running, and are you in the 'docker' group?" >&2
    echo "            (sudo usermod -aG docker \$USER, then log out and back in)" >&2
    exit 1; }

mkdir -p "$HOME/.local/bin"
install -m 755 dobotpbl "$HOME/.local/bin/dobotpbl"
echo "installed dobotpbl -> $HOME/.local/bin/dobotpbl"

# The launcher defaults to ~/Software/Coding/dobot-lab; point it here instead if
# this clone lives somewhere else.
if [ "$PROJECT" != "$HOME/Software/Coding/dobot-lab" ]; then
    echo
    echo "This clone is at $PROJECT, which is not the launcher's default."
    echo "Add this to your ~/.bashrc so dobotpbl finds it:"
    echo
    echo "    export DOBOTPBL_PROJECT=\"$PROJECT\""
    echo
fi

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "NOTE: $HOME/.local/bin is not on your PATH. Add to ~/.bashrc:"
       echo '    export PATH="$HOME/.local/bin:$PATH"'; echo ;;
esac

echo "building the image (first time takes a few minutes) ..."
docker build -t dobot-lab:latest \
    --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" "$PROJECT"

echo
echo "done. Try:  DOBOTPBL_PROJECT=\"$PROJECT\" dobotpbl --test"
