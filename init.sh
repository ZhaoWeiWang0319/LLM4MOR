#!/usr/bin/env bash
# Configure and check tools for llm_analyze.py and verify_solution.py.
# Usage: source init.sh [--ampl PATH] [--minizinc PATH] [--python PATH] [--gurobi DIR] [--highs PATH]

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run with source so tool paths remain in this shell:" >&2
    echo "  source init.sh [--ampl PATH] [--minizinc PATH] [--python PATH] [--gurobi DIR] [--highs PATH]" >&2
    exit 1
fi

set -u
LLM4MOR_INIT_ORIGINAL_DIR="$PWD"
llm4mor_restore_directory() {
    local saved_dir="$LLM4MOR_INIT_ORIGINAL_DIR"
    unset LLM4MOR_INIT_ORIGINAL_DIR
    cd "$saved_dir"
}
cd "$(dirname "${BASH_SOURCE[0]}")"

ok()   { echo "  [OK] $*"; }
warn() { echo "  [!!] $*"; }

AMPL_CMD="${LLM4MOR_AMPL:-ampl}"
MINIZINC_CMD="${LLM4MOR_MINIZINC:-minizinc}"
PYTHON="${LLM4MOR_PYTHON:-${PYTHON:-python3}}"
GUROBI_DIR="${LLM4MOR_GUROBI:-}"
HIGHS_CMD="${LLM4MOR_HIGHS:-highs}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ampl|--minizinc|--python|--gurobi|--highs)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                llm4mor_restore_directory || return 1
                unset -f llm4mor_restore_directory
                return 2
            fi
            case "$1" in
                --ampl) AMPL_CMD="$2" ;;
                --minizinc) MINIZINC_CMD="$2" ;;
                --python) PYTHON="$2" ;;
                --gurobi) GUROBI_DIR="$2" ;;
                --highs) HIGHS_CMD="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            echo "Usage: source init.sh [--ampl PATH] [--minizinc PATH] [--python PATH] [--gurobi DIR] [--highs PATH]"
            llm4mor_restore_directory || return 1
            unset -f llm4mor_restore_directory
            return 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: source init.sh [--ampl PATH] [--minizinc PATH] [--python PATH] [--gurobi DIR] [--highs PATH]" >&2
            llm4mor_restore_directory || return 1
            unset -f llm4mor_restore_directory
            return 2
            ;;
    esac
done

GUROBI_CMD="${GUROBI_DIR:+$GUROBI_DIR/bin/}gurobi_cl"

export LLM4MOR_AMPL="$AMPL_CMD"
export LLM4MOR_MINIZINC="$MINIZINC_CMD"
export LLM4MOR_PYTHON="$PYTHON"
export LLM4MOR_GUROBI="$GUROBI_DIR"
export LLM4MOR_HIGHS="$HIGHS_CMD"

echo ">> Runtime tools"
if command -v "$PYTHON" >/dev/null 2>&1; then
    ok "Python: $(command -v "$PYTHON")"
    "$PYTHON" -c 'import sys; print("       Version:", sys.version.split()[0])'
else
    warn "Python not found: $PYTHON. Set LLM4MOR_PYTHON or add python3 to PATH."
fi

if command -v "$AMPL_CMD" >/dev/null 2>&1; then
    ok "AMPL: $(command -v "$AMPL_CMD")"
else
    warn "AMPL not found: $AMPL_CMD. Set LLM4MOR_AMPL or add ampl to PATH (required for .mod files)."
fi

if command -v "$MINIZINC_CMD" >/dev/null 2>&1; then
    ok "MiniZinc: $(command -v "$MINIZINC_CMD")"
else
    warn "MiniZinc not found: $MINIZINC_CMD. Set LLM4MOR_MINIZINC or add minizinc to PATH (required for .mzn files)."
fi

if command -v "$GUROBI_CMD" >/dev/null 2>&1; then
    ok "Gurobi: $(command -v "$GUROBI_CMD")"
    "$GUROBI_CMD" --version 2>/dev/null | sed 's/^/       /'
else
    warn "Gurobi not found: $GUROBI_CMD. Set LLM4MOR_GUROBI or add gurobi_cl to PATH."
fi

if command -v "$HIGHS_CMD" >/dev/null 2>&1; then
    ok "HiGHS: $(command -v "$HIGHS_CMD")"
    "$HIGHS_CMD" --version 2>/dev/null | sed 's/^/       /'
else
    warn "HiGHS not found: $HIGHS_CMD. Set LLM4MOR_HIGHS or add highs to PATH."
fi

echo
echo ">> Project tool environment"
echo "       LLM4MOR_AMPL=$LLM4MOR_AMPL"
echo "       LLM4MOR_MINIZINC=$LLM4MOR_MINIZINC"
echo "       LLM4MOR_PYTHON=$LLM4MOR_PYTHON"
echo "       LLM4MOR_GUROBI=$LLM4MOR_GUROBI"
echo "       LLM4MOR_HIGHS=$LLM4MOR_HIGHS"

echo
echo ">> LLM API"
if [ -n "${LLM_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
    ok "An API key is set"
else
    warn "No API key is set. Use your provider's credentials."
fi
echo "       LLM_BASE_URL=${LLM_BASE_URL:-https://api.openai.com/v1}"
echo "       LLM_MODEL=${LLM_MODEL:-<model-name>}"

cat <<'EOF'

>> Configure an OpenAI-compatible API
   export LLM_API_KEY="your-api-key"
   export LLM_BASE_URL="https://your-provider.example/v1"
   export LLM_MODEL="your-model-name"

>> Run
   $LLM4MOR_PYTHON llm_analyze.py <model.mod-or-mzn> --output-dir <output-directory>
   $LLM4MOR_PYTHON verify_solution.py <original-model> --sol <solution.sol.json> --map <mapping.py>

   bash example/verify.sh verifies the supplied example solution.
EOF

if ! llm4mor_restore_directory; then
    echo "Could not restore the initial working directory." >&2
    unset -f llm4mor_restore_directory
    return 1
fi
unset -f llm4mor_restore_directory
