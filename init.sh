#!/usr/bin/env bash
# init.sh
# ─────────────────────────────────────────────────────────────────────────────
# Environment bootstrap for the Academic Performance Modelling project.
# Run this at the start of every session before touching enrich.py or verify.py.
#
# Usage:
#   bash init.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║  Academic Performance Modelling — Environment Bootstrap      ║"
echo "  ║  Regression & Time Series · Mountain Top University · 2026   ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ─── 1. Python version check ──────────────────────────────────────────────────
echo -e "${CYAN}[1/5] Checking Python version...${RESET}"
PYTHON=$(command -v python3 || command -v python || echo "")
if [ -z "$PYTHON" ]; then
  echo -e "${RED}  ERROR: Python not found. Install Python >= 3.9.${RESET}"
  exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
  echo -e "${RED}  ERROR: Python $PY_VERSION found, but >= 3.9 required.${RESET}"
  exit 1
fi
echo -e "${GREEN}  Python $PY_VERSION — OK${RESET}"

# ─── 2. Install / verify dependencies ────────────────────────────────────────
echo -e "${CYAN}[2/5] Checking dependencies...${RESET}"

DEPS=("numpy" "pandas" "scipy" "sklearn")
MISSING=()

for dep in "${DEPS[@]}"; do
  if ! $PYTHON -c "import $dep" 2>/dev/null; then
    MISSING+=("$dep")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo -e "${YELLOW}  Missing packages: ${MISSING[*]}. Installing...${RESET}"
  # Map sklearn back to scikit-learn for pip
  PIP_PKGS=()
  for pkg in "${MISSING[@]}"; do
    if [ "$pkg" = "sklearn" ]; then
      PIP_PKGS+=("scikit-learn")
    else
      PIP_PKGS+=("$pkg")
    fi
  done
  $PYTHON -m pip install --quiet "${PIP_PKGS[@]}"
  echo -e "${GREEN}  Installed: ${PIP_PKGS[*]}${RESET}"
else
  echo -e "${GREEN}  All dependencies present — OK${RESET}"
fi

# ─── 3. Verify required files are present ────────────────────────────────────
echo -e "${CYAN}[3/5] Checking required files...${RESET}"

REQUIRED_FILES=(
  "academic_performance_dataset_V2.csv"
  "enrich.py"
  "verify.py"
  "README.md"
  "AGENTS.md"
  "memory.md"
  "context.md"
)

ALL_PRESENT=true
for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "$f" ]; then
    echo -e "${GREEN}  ✓  $f${RESET}"
  else
    echo -e "${RED}  ✗  $f  ← MISSING${RESET}"
    ALL_PRESENT=false
  fi
done

if [ "$ALL_PRESENT" = false ]; then
  echo -e "${RED}  ERROR: One or more required files are missing. Cannot proceed.${RESET}"
  exit 1
fi

# ─── 4. Source data integrity check ──────────────────────────────────────────
echo -e "${CYAN}[4/5] Validating source data...${RESET}"

$PYTHON - <<'PYEOF'
import sys
import pandas as pd

path = "academic_performance_dataset_V2.csv"
df = pd.read_csv(path)

required_cols = {"ID No","Prog Code","Gender","YoG","CGPA",
                 "CGPA100","CGPA200","CGPA300","CGPA400","SGPA"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    print(f"\033[31m  ERROR: Source file missing columns: {missing_cols}\033[0m")
    sys.exit(1)

if len(df) < 100:
    print(f"\033[31m  ERROR: Source file has only {len(df)} rows — suspiciously small.\033[0m")
    sys.exit(1)

if df.isnull().sum().sum() > 0:
    print(f"\033[33m  WARN: Source file has {df.isnull().sum().sum()} null values.\033[0m")
else:
    print(f"\033[32m  Source data: {len(df)} rows × {len(df.columns)} columns, no nulls — OK\033[0m")
PYEOF

# ─── 5. Run enrich + verify pipeline ─────────────────────────────────────────
echo -e "${CYAN}[5/5] Running enrich → verify pipeline...${RESET}"
echo ""

$PYTHON enrich.py \
  --input  academic_performance_dataset_V2.csv \
  --output academic_performance_enriched.csv \
  --seed   42

echo ""
$PYTHON verify.py \
  --input academic_performance_enriched.csv \
  --alpha 0.05

VERIFY_EXIT=$?

echo ""
if [ $VERIFY_EXIT -eq 0 ]; then
  echo -e "${GREEN}"
  echo "  ╔══════════════════════════════════════════════════════════════╗"
  echo "  ║  Environment ready. All checks passed.                       ║"
  echo "  ║  academic_performance_enriched.csv is ready for modelling.   ║"
  echo "  ╚══════════════════════════════════════════════════════════════╝"
  echo -e "${RESET}"
  exit 0
else
  echo -e "${RED}"
  echo "  ╔══════════════════════════════════════════════════════════════╗"
  echo "  ║  PIPELINE FAILED — do not proceed to modelling.             ║"
  echo "  ║  Review FAIL lines in verify.py output above.               ║"
  echo "  ╚══════════════════════════════════════════════════════════════╝"
  echo -e "${RESET}"
  exit 1
fi
