#!/bin/bash

for f in results/*; do
    if [[ ! -d "$f/pytorch_model_0" ]]; then
        echo rm -rf $f
        rm -rf $f
        echo rm -rf "logs/$(basename $f)"
        rm -rf "logs/$(basename $f)"
    fi
done
