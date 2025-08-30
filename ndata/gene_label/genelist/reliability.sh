#!/bin/bash

# extract different reliability genelist form subcelluar_location.csv

# 生成各个标签可靠性的基因序列

input=subcellular_location.tsv
enhanced=enhanced.txt
supported=supported.txt
approved=approved.txt
uncertain=uncertain.txt

awk '$3 == "Enhanced" {print $1}' $input | sort | uniq > $enhanced
awk '$3 == "Supported" {print $1}' $input | sort | uniq > $supported
awk '$3 == "Approved" {print $1}' $input | sort | uniq > $approved
awk '$3 == "Uncertain" {print $1}' $input | sort | uniq > $uncertain
