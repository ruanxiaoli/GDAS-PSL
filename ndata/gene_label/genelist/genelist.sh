#!/bin/bash
# 生成总的基因
# generate gene list from subcellular_location.tsv

#input=normal_tissue.tsv
input=subcellular_location.tsv

output=genelist.txt

awk 'NR > 1 {print $1}' $input | sort | uniq > $output
