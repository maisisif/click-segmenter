#!/bin/bash
# One-time environment setup on a MetaCentrum frontend (e.g. tarkil.metacentrum.cz).
# Run this yourself over SSH -- it is NOT a PBS job (frontends are fine for this
# kind of light setup work, just not for actual computing).
#
#   ssh <username>@tarkil.metacentrum.cz
#   bash setup_env.sh
#
# What it does:
#   1. Clones (or updates) click-segmenter into ~/projects, mirroring the local
#      dev layout so configs/data.yaml's "~/projects/ade20k-reference/..." path
#      resolves the same way on the cluster as it does locally.
#   2. Expects ade20k-reference (the CSAILVision/ADE20K toolkit repo with the
#      3-image sample) to be cloned alongside it -- do that manually first if
#      you haven't:
#        cd ~/projects && git clone https://github.com/CSAILVision/ADE20K.git ade20k-reference
#      (adjust the URL/path if your local clone came from somewhere else --
#      just make sure the resulting layout matches what's in configs/data.yaml)
#   3. Creates a persistent venv at ~/venvs/click-segmenter and installs
#      requirements.txt into it.
#
# Module name note: MetaCentrum's Python module is called `python-modules`
# (see https://docs.metacentrum.cz/en/docs/software/modules). Run
# `module avail python` yourself first if this has changed -- module names on
# HPC systems get renamed/versioned over time.

set -euo pipefail

REPO_URL="https://github.com/maisisif/click-segmenter.git"
PROJECTS_DIR="$HOME/projects"
VENV_DIR="$HOME/venvs/click-segmenter"

mkdir -p "$PROJECTS_DIR"
cd "$PROJECTS_DIR"

if [ -d click-segmenter/.git ]; then
    echo "click-segmenter already cloned, pulling latest..."
    (cd click-segmenter && git pull)
else
    git clone "$REPO_URL" click-segmenter
fi

if [ ! -d ade20k-reference ]; then
    echo "WARNING: $PROJECTS_DIR/ade20k-reference not found."
    echo "The hello-GPU job needs it for the bundled 3-image sample."
    echo "Clone or scp it here before submitting the job -- see comment above."
fi

module avail python 2>&1 | head -20
module load python-modules

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECTS_DIR/click-segmenter/requirements.txt"

echo "Done. Venv ready at $VENV_DIR"
echo "Next: qsub ~/projects/click-segmenter/scripts/metacentrum/hello_gpu.pbs"
