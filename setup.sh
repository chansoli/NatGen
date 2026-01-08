#!/bin/bash

ENV_NAME=${ENV_NAME:-natgen}

function ensure_env_exists() {
    if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1; then
        echo "ERROR: conda env '$ENV_NAME' not found. Run create_and_activate first."
        exit 1
    fi
}

function get_env_prefix() {
    ensure_env_exists
    conda run -n "$ENV_NAME" python -c 'import os, sys; prefix = os.environ.get("CONDA_PREFIX") or sys.prefix; print(prefix, end="")'
}

function setup_repo() {
    ensure_env_exists
    mkdir -p sitter-libs

    # Optional: clean old clones so rerunning doesn’t whine
    # rm -rf sitter-libs/* parser/languages.so

    git clone https://github.com/tree-sitter/tree-sitter-go         sitter-libs/go
    git clone https://github.com/tree-sitter/tree-sitter-javascript sitter-libs/js
    git clone https://github.com/tree-sitter/tree-sitter-c          sitter-libs/c
    git clone https://github.com/tree-sitter/tree-sitter-cpp        sitter-libs/cpp
    git clone https://github.com/tree-sitter/tree-sitter-c-sharp    sitter-libs/cs
    git clone https://github.com/tree-sitter/tree-sitter-python     sitter-libs/py
    git clone https://github.com/tree-sitter/tree-sitter-java       sitter-libs/java
    git clone https://github.com/tree-sitter/tree-sitter-ruby       sitter-libs/ruby
    git clone https://github.com/tree-sitter/tree-sitter-php        sitter-libs/php

    # --- Pin each grammar to an older (NatGen-era) tag ---
    # Try v0.19.0 first, then 0.19.0 (some repos drop the leading 'v').
    (cd sitter-libs/go   && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/js   && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/c    && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/cpp  && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/cs   && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/py   && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/java && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/ruby && (git checkout v0.19.0   || git checkout 0.19.0   || true))
    (cd sitter-libs/php  && (git checkout v0.19.0   || git checkout 0.19.0   || true))

    mkdir -p parser

    local env_prefix
    env_prefix="$(get_env_prefix)"

    CC="$env_prefix/bin/x86_64-conda-linux-gnu-gcc" \
    CXX="$env_prefix/bin/x86_64-conda-linux-gnu-g++" \
    conda run -n "$ENV_NAME" python create_tree_sitter_parser.py sitter-libs

    cp parser/languages.so src/evaluator/CodeBLEU/parser/languages.so
}

function create_and_activate() {
    conda create --name "$ENV_NAME" python=3.8 -y
    echo "Environment '$ENV_NAME' created. Activate it manually if you need an interactive shell."
}

function install_deps() {
    ensure_env_exists
    local env_name="$ENV_NAME"

    # IMPORTANT: Do NOT install conda-forge "compilers" / gxx_linux-64 here.
    # They bring in compiler_compat/ld which caused the /lib64/libc.so.6 error.

    # If you haven't already, outside conda (once):
    #   sudo apt-get update
    #   sudo apt-get install -y build-essential

    # ---- PyTorch stack (fixed) ----
    conda install -n "$env_name" -y \
        pytorch==1.7.0 \
        torchvision \
        torchaudio \
        cudatoolkit=11.1 \
        -c pytorch -c conda-forge

    # ---- HuggingFace stack (fixed) ----
    conda install -n "$env_name" -y -c conda-forge \
        datasets==1.18.3 \
        transformers==4.16.2 \
        tensorboard==2.8.0 \
        sentencepiece        # needed for Salesforce/codet5-base

    conda install -n "$env_name" -y -c conda-forge mkl mkl-service

    # ---- Make sure pip tooling is recent ----
    conda run -n "$env_name" python -m pip install --upgrade pip setuptools wheel

    # ---- pip packages (all fixed) ----
    # Use system GCC/LD, not conda's compiler_compat
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++

    conda run -n "$env_name" pip install "huggingface_hub==0.13.4" --force-reinstall
    conda run -n "$env_name" pip install tree-sitter==0.19.0
    conda run -n "$env_name" pip install nltk==3.6.7
    conda run -n "$env_name" pip install scipy==1.5.4
    conda run -n "$env_name" pip install pytest

    local env_prefix
    env_prefix="$(get_env_prefix)"

    if [[ -z "$env_prefix" ]]; then
        echo "ERROR: Unable to determine CONDA_PREFIX for $env_name"
        exit 1
    fi

    local hf_home="$env_prefix/hf_home"
    mkdir -p "$hf_home"

    # LD_PRELOAD required for language.so loading
    # HF_HOME required to use local cache
    conda env config vars set --name "$env_name" \
        LD_PRELOAD="$env_prefix/lib/libstdc++.so.6" \
        HF_HOME="$hf_home"

    # Please add the command if you add any package.
}

function setup_hf() {
    # Mirror the Salesforce/codet5-base weights into the HF cache within the conda env
    ensure_env_exists
    local hf_script
    hf_script=$(cat <<'PY'
import os
from huggingface_hub import snapshot_download

hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise SystemExit("HF_HOME is not configured. Run install_deps to set it up.")

print(f"Downloading Salesforce/codet5-base into {hf_home}")
snapshot_download(
    repo_id="Salesforce/codet5-base",
    cache_dir=hf_home,
    resume_download=True,
    local_files_only=False,
)
PY
)

    conda run -n "$ENV_NAME" python -c "$hf_script"
}

create_and_activate;
install_deps;
setup_repo;
setup_hf;
