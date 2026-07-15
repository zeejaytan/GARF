#!/bin/bash

echo "Monitoring fracture test jobs..."
echo "================================"

# List our specific jobs
squeue -u $USER | grep -E "(13348784|13348790|13348791|13348861|13348915)"

echo ""
echo "All user jobs:"
squeue -u $USER

echo ""
echo "To check job status in detail, use:"
echo "  squeue -j <job_id>"
echo "  scontrol show job <job_id>"
echo ""
echo "To check output/error files:"
echo "  cat quick_fracture_test_<job_id>.out"
echo "  cat quick_fracture_test_<job_id>.err"
echo "  cat tray_fracture_test_<job_id>.out"
echo "  cat tray_fracture_test_<job_id>.err"
echo "  cat tray000_fracture_test_<job_id>.out"
echo "  cat tray000_fracture_test_<job_id>.err"