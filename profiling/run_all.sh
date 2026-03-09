#!/bin/bash

# Comprehensive profiling script for Kineto vs NSight Compute comparison
# Profiles both Config A and Config B with both profilers

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Kineto vs NVIDIA NSight Compute - Comprehensive Profiling Suite${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. CUDA is required.${NC}"
    exit 1
fi

echo -e "${GREEN}GPU Information:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check for ncu
NCU_AVAILABLE=false
if command -v ncu &> /dev/null; then
    NCU_AVAILABLE=true
    echo -e "${GREEN}NVIDIA NSight Compute found:${NC} $(ncu --version | head -n 1)"
else
    echo -e "${YELLOW}WARNING: ncu not found. NSight Compute profiling will be skipped.${NC}"
    echo -e "${YELLOW}To install: https://developer.nvidia.com/nsight-compute${NC}"
fi
echo ""

# Parse arguments
SKIP_KINETO=false
SKIP_NSIGHT=false
CONFIGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-kineto)
            SKIP_KINETO=true
            shift
            ;;
        --skip-nsight)
            SKIP_NSIGHT=true
            shift
            ;;
        --config)
            CONFIGS+=("$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-kineto       Skip Kineto profiling"
            echo "  --skip-nsight       Skip NSight Compute profiling"
            echo "  --config <path>     Profile specific config (can be used multiple times)"
            echo "                      Default: profiles both config_a.yaml and config_b.yaml"
            echo "  --help              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no configs specified, use default configs
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(
        "$PROJECT_ROOT/configs/config_a.yaml"
        "$PROJECT_ROOT/configs/config_b.yaml"
    )
fi

# Validate config files exist
for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo -e "${RED}ERROR: Config file not found: $config${NC}"
        exit 1
    fi
done

echo -e "${BLUE}Profiling Configuration:${NC}"
echo -e "  Configs to profile: ${#CONFIGS[@]}"
for config in "${CONFIGS[@]}"; do
    echo -e "    - $(basename $config)"
done
echo -e "  Kineto profiling: $([ "$SKIP_KINETO" = false ] && echo "${GREEN}ENABLED${NC}" || echo "${YELLOW}SKIPPED${NC}")"
echo -e "  NSight profiling: $([ "$SKIP_NSIGHT" = false ] && [ "$NCU_AVAILABLE" = true ] && echo "${GREEN}ENABLED${NC}" || echo "${YELLOW}SKIPPED${NC}")"
echo -e "  Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Start profiling
START_TIME=$(date +%s)
FAILED_RUNS=()

for config in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$config" .yaml)

    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  Profiling: $CONFIG_NAME${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo ""

    # Kineto profiling
    if [ "$SKIP_KINETO" = false ]; then
        echo -e "${GREEN}[1/2] Running Kineto profiling for $CONFIG_NAME...${NC}"
        if python "$SCRIPT_DIR/profile_kineto.py" \
            --config "$config" \
            --output-dir "$RESULTS_DIR"; then
            echo -e "${GREEN}✓ Kineto profiling completed for $CONFIG_NAME${NC}"
        else
            echo -e "${RED}✗ Kineto profiling failed for $CONFIG_NAME${NC}"
            FAILED_RUNS+=("Kineto-$CONFIG_NAME")
        fi
        echo ""
    fi

    # NSight Compute profiling
    if [ "$SKIP_NSIGHT" = false ] && [ "$NCU_AVAILABLE" = true ]; then
        echo -e "${GREEN}[2/2] Running NSight Compute profiling for $CONFIG_NAME...${NC}"
        echo -e "${YELLOW}Note: This may take significantly longer than Kineto (10-100x slowdown)${NC}"
        if python "$SCRIPT_DIR/profile_nsight.py" \
            --config "$config" \
            --output-dir "$RESULTS_DIR"; then
            echo -e "${GREEN}✓ NSight Compute profiling completed for $CONFIG_NAME${NC}"
        else
            echo -e "${RED}✗ NSight Compute profiling failed for $CONFIG_NAME${NC}"
            FAILED_RUNS+=("NSight-$CONFIG_NAME")
        fi
        echo ""
    fi

    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Profiling Summary${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "Total time: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
echo -e "Results saved to: $RESULTS_DIR"
echo ""

if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All profiling runs completed successfully!${NC}"
else
    echo -e "${YELLOW}Some profiling runs failed:${NC}"
    for failed in "${FAILED_RUNS[@]}"; do
        echo -e "  ${RED}✗${NC} $failed"
    done
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. View Kineto traces in Chrome:"
echo -e "     ${YELLOW}chrome://tracing${NC} and load ${YELLOW}results/kineto_*/trace.json${NC}"
echo ""
echo -e "  2. View NSight Compute reports:"
echo -e "     ${YELLOW}ncu-ui results/nsight_*/nsight_*.ncu-rep${NC}"
echo ""
echo -e "  3. Generate comparison analysis:"
echo -e "     ${YELLOW}python analysis/compare_results.py${NC}"
echo ""
echo -e "${BLUE}======================================================================${NC}"
