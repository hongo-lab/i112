#!/bin/sh

if ls $1; then
    echo $1 exists
else
    echo $1 dose not exist
fi

