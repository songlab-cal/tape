#!/bin/bash

for f in results/*; do
    if [[ ! `compgen -G "$f/pytorch_model*"` ]]; then
        echo rm -rf $f
        rm -rf $f
        echo rm -rf "logs/$(basename $f)"
        rm -rf "logs/$(basename $f)"
    fi
done
