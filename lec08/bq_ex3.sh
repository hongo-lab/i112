#!/bin/sh

VAR=`date '+%Y-%m-%d-%H-%M'`
touch ${VAR}.log

i=1
echo "$i + 1"
echo `expr $i + 1`