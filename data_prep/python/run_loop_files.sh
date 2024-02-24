#!/bin/bash
for i in {1..10}
do
    echo $i
    python3 loop_files.py $i
done