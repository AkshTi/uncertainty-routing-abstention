#!/bin/bash
#
# Check All Results
# Verifies that all experiments completed successfully
#

echo "========================================================================"
echo " RESULTS VERIFICATION"
echo "========================================================================"
echo ""

echo "Checking experiment summaries..."
echo ""

all_complete=true

for i in {1..9}; do
    file="results/exp${i}_summary.json"
    if [ -f "$file" ]; then
        echo "✓ Experiment $i: COMPLETE"
    else
        echo "✗ Experiment $i: MISSING"
        all_complete=false
    fi
done

echo ""
echo "Checking critical files..."
echo ""

if [ -f "results/steering_vectors_explicit.pt" ]; then
    echo "✓ Steering vectors: PRESENT"
else
    echo "✗ Steering vectors: MISSING"
    all_complete=false
fi

if [ -f "results/complete_pipeline_standard.json" ]; then
    echo "✓ Combined summary: PRESENT"
else
    echo "⚠ Combined summary: MISSING (run full pipeline to generate)"
fi

echo ""
echo "Checking figures..."
echo ""

figure_count=$(ls results/*.png 2>/dev/null | wc -l)
echo "Found $figure_count figure(s)"

if [ $figure_count -ge 9 ]; then
    echo "✓ All figures generated"
else
    echo "⚠ Some figures may be missing"
fi

echo ""
echo "========================================================================"

if [ "$all_complete" = true ]; then
    echo " ✓ ALL EXPERIMENTS COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. View summaries: cat results/exp8_summary.json"
    echo "  2. Check figures: open results/*.png"
    echo "  3. Extract numbers for paper (see PAPER_OUTLINE.md)"
    echo ""
else
    echo " ✗ SOME EXPERIMENTS INCOMPLETE"
    echo "========================================================================"
    echo ""
    echo "Run the missing segments:"
    echo "  ./run_segment1.sh  (Exp1-3)"
    echo "  ./run_segment2.sh  (Exp4-5)"
    echo "  ./run_segment3.sh  (Exp6-7)"
    echo "  ./run_segment4.sh  (Exp8-9) ⭐ CRITICAL"
    echo ""
fi
