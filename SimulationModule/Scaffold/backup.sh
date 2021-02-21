#!/bin/bash
date=`date +%Y_%m_%d_%H_%M`
mkdir ./backup/"$1_${date}"

cp *.cu *.h Makefile ./backup/"$1_${date}"
