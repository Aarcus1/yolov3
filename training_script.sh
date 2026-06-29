#!/bin/bash

max_retries=1000
count=0

while true; do
  count=$((count+1))

  if [ $count -ge $max_retries ]; then
    echo "Reached max retries ($max_retries). Exiting..."
    exit 1
  fi
  source_base="./configurations/"
  source_path=$(realpath "$source_base")
  export PYTHONPATH="$source_path:$PYTHONPATH"

  python  source/train.py --config_folder ./configurations/
  status=$?

  if [ $status -eq 0 ]; then
    echo "Training finished successfully"
    break
  else
    echo "Training crashed with exit code $status. Retrying..."
    sleep 5
  fi
done