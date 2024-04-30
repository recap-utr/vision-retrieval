#!/bin/bash
apptainer run --nv ~/poetry.sif run jupyter lab --allow-root --no-browser --ip 0.0.0.0 --port 6789 --notebook-dir=..
