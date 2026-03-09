#!/bin/bash

# Run All Profilers Script
#
# This script runs both Kineto and NVIDIA NSight Compute profiling
# for specified configurations.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="results"
CONFIGS=("config_a" "config_b")

# Parse arguments
PROFILERS=("kineto" "nsight")
RUN_KINETO=true
RUN_NSIGHT=true
RUN_ANALYSIS=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -k, --kineto-only     Run only Kineto profiling"
    echo "  -n, --nsight-only     Run only NSight profiling"
    echo "  -s, --skip-analysis   Skip comparison analysis"
    echo "  -c, --configs CONFIG  Comma-separated config names (default: config_a,config_b)"
    echo "  -o, --output DIR      Output directory (default: results)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all profilers on all configs"
    echo "  $0 --kineto-only             # Run only Kineto"
    echo "  $0 -c config_a               # Run only config_a"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--kineto-only)
            RUN_NSIGHT=false
            shift
            ;;
        -n|--nsight-only)
            RUN_KINETO=false
            shift
            ;;
        -s|--skip-analysis)
            RUN_ANALYSIS=false
            shift
            ;;
        -c|--configs)
            IFS=',' read -ra CONFIGS <<< "$2"
            shift 2
            ;;
        -o|--output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GPU Profiling Benchmark Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configurations: ${CONFIGS[@]}"
echo "Output directory: $RESULTS_DIR"
echo "Kineto: $RUN_KINETO"
echo "NSight: $RUN_NSIGHT"
echo "Analysis: $RUN_ANALYSIS"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run profiling for each configuration
for config in "${CONFIGS[@]}"; do
    config_file="configs/${config}.yaml"

    if [ ! -f "$config_file" ]; then
        echo -e "${RED}ERROR: Config file not found: $config_file${NC}"
        continue
    fi

    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Processing: $config${NC}"
    echo -e "${YELLOW}========================================${NC}\n"

    # Run Kineto profiling
    if [ "$RUN_KINETO" = true ]; then
        echo -e "${GREEN}Running Kineto profiling...${NC}"
        python3 profiling/profile_kineto.py \
            --config "$config_file" \
            --output-dir "$RESULTS_DIR"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Kineto profiling complete${NC}"
        else
            echo -e "${RED}✗ Kineto profiling failed${NC}"
        fi
        echo ""
    fi

    # Run NSight profiling
    if [ "$RUN_NSIGHT" = true ]; then
        echo -e "${GREEN}Running NVIDIA NSight Compute profiling...${NC}"
        python3 profiling/profile_nsight.py \
            --config "$config_file" \
            --output-dir "$RESULTS_DIR"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ NSight profiling complete${NC}"
        else
            echo -e "${RED}✗ NSight profiling failed (may need to run manually)${NC}"
        fi
        echo ""
    fi
done

# Run comparison analysis
if [ "$RUN_ANALYSIS" = true ]; then
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Running Comparison Analysis${NC}"
    echo -e "${YELLOW}========================================${NC}\n"

    python3 analysis/compare_profilers.py \
        --configs "${CONFIGS[@]}" \
        --results-dir "$RESULTS_DIR" \
        --output "$RESULTS_DIR/profiler_comparison_report.md"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Comparison analysis complete${NC}"
    else
        echo -e "${RED}✗ Comparison analysis failed${NC}"
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  All profiling complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "View results:"
echo "  - Kineto Chrome traces: open chrome://tracing and load ${RESULTS_DIR}/kineto_*/kineto_trace_*.json"
echo "  - NSight reports: ncu-ui ${RESULTS_DIR}/nsight_*/nsight_*.ncu-rep"
echo "  - Comparison report: cat ${RESULTS_DIR}/profiler_comparison_report.md"
echo ""
