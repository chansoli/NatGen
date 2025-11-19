#!/bin/bash

function setup_repo() {
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

    CC=x86_64-conda-linux-gnu-gcc \
    CXX=x86_64-conda-linux-gnu-g++ \
    python3 create_tree_sitter_parser.py sitter-libs

    cp parser/languages.so src/evaluator/CodeBLEU/parser/languages.so
}

function create_and_activate() {
    conda create --name natgen python=3.6;
    conda activate natgen;
}

function install_deps() {
    # ---- C/C++ toolchain (for tree-sitter, static_assert, C11, etc.) ----
    conda install -y -c conda-forge compilers gxx_linux-64 libcxx libstdcxx-ng

    # ---- PyTorch stack ----
    #conda install -y pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
    conda install -y pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

    # ---- HuggingFace dependencies ----
    conda install -y datasets==1.18.3 -c conda-forge
    conda install -y transformers==4.16.2 -c conda-forge
    conda install -y tensorboard==2.8.0 -c conda-forge

    # ---- pip packages ----
    pip install tree-sitter==0.19.0
    pip install nltk==3.6.7
    pip install scipy==1.5.4

    # Please add the command if you add any package.
}

# create_and_activate;
# install_deps;
setup_repo;

# Before running scripts using language.so
# export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
