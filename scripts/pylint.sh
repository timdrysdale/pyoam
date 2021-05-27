#!/bin/bash
cd  ../src/pyoam
pylint --good-names=A,f,i,k,r,v,x,y,z,_ --variable-rgx='^[a-z][a-z0-9]*((_[a-z0-9]+)*)?$' --argument-rgx='^[a-z][a-z0-9]*((_[a-z0-9]+)*)?$' ./*
cd ../../scripts

