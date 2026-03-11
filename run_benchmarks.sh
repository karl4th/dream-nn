#!/bin/bash
# DREAM Benchmark Runner
# Run all 5 benchmark tests with a single command

# Default values
AUDIO_DIR="${AUDIO_DIR:-audio_test}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"

echo "======================================================================"
echo "DREAM Benchmark Suite - All 5 Tests"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Audio Directory: $AUDIO_DIR"
echo "  Device: $DEVICE"
echo "  Epochs: $EPOCHS"
echo ""
echo "======================================================================"
echo ""

# Run all benchmarks
uv run python -m dream.benchmarks.run_all_benchmarks \
    --audio-dir "$AUDIO_DIR" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    "$@"

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All benchmarks completed successfully!"
else
    echo "⚠️  Some benchmarks failed. Check the report for details."
fi
echo "======================================================================"
echo ""
echo "Results saved to: tests/benchmarks/results/"
echo "  - benchmark_summary.json"
echo "  - BENCHMARK_REPORT.md"
echo ""

exit $EXIT_CODE
