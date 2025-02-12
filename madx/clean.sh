#!/bin/bash

# This script moves all MAD-X output files to a new "outputs" directory.
outdir="outputs";
script=$(basename $BASH_SOURCE);

mkdir -p $outdir

for file in *; do
  if [[ (! -d $file) && ($file != $script) && ($file != *.mad) && ($file != *.lat) ]]; then
    mv $file $outdir
  fi
done
