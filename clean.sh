#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=${ENV_NAME:-natgen}
FORCE=false
KEEP_ENV=false
KEEP_HF_CACHE=false

usage() {
    cat <<'EOF'
Usage: ./clean.sh [options]

Removes artifacts created by setup.sh.

Options:
  -y, --yes           Proceed without confirmation.
  --keep-env          Do not delete the conda environment.
  --keep-hf-cache     Keep the HF_HOME cache directory.
  -h, --help          Show this message.

Environment variables:
  ENV_NAME            Conda environment to clean (default: natgen).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            FORCE=true
            ;;
        --keep-env)
            KEEP_ENV=true
            ;;
        --keep-hf-cache)
            KEEP_HF_CACHE=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

env_exists() {
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1
}

get_env_prefix() {
    conda run -n "$ENV_NAME" python -c 'import os, sys; print(os.environ.get("CONDA_PREFIX") or sys.prefix, end="")'
}

remove_path() {
    local target="$1"
    local label="$2"
    if [[ -e "$target" ]]; then
        rm -rf "$target"
        echo "Removed $label: $target"
    else
        echo "No $label to remove at $target"
    fi
}

if [[ "$FORCE" = false ]]; then
    echo "This will delete sitter-libs/, parser/, parser artifacts, and the '$ENV_NAME' conda environment.";
    read -r -p "Continue? [y/N] " reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        echo "Abort. Nothing removed."
        exit 0
    fi
fi

remove_path "$SCRIPT_DIR/sitter-libs" "tree-sitter grammars"
remove_path "$SCRIPT_DIR/parser" "parser build output"
remove_path "$SCRIPT_DIR/src/evaluator/CodeBLEU/parser/languages.so" "CodeBLEU parser binary"

if env_exists; then
    env_prefix="$(get_env_prefix)"

    if [[ "$KEEP_ENV" = true ]]; then
        if [[ "$KEEP_HF_CACHE" = false ]]; then
            remove_path "$env_prefix/hf_home" "HF cache"
        else
            echo "Keeping HF cache at $env_prefix/hf_home"
        fi

        echo "Removing LD_PRELOAD/HF_HOME env vars from $ENV_NAME"
        conda env config vars unset --name "$ENV_NAME" LD_PRELOAD HF_HOME 2>/dev/null || true
    else
        if [[ "$KEEP_HF_CACHE" = false ]]; then
            remove_path "$env_prefix/hf_home" "HF cache"
        else
            echo "Keeping HF cache at $env_prefix/hf_home"
        fi

        echo "Removing conda env '$ENV_NAME'"
        conda env remove --name "$ENV_NAME" -y
    fi
else
    echo "Conda env '$ENV_NAME' not found; skipping env removal."
fi

echo "Cleanup complete."
