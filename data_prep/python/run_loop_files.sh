#!/bin/bash
for i in {1..10}
do
    echo $i
    python loop_files.py $i
done