#!/bin/bash
pylint --good-names=A,f,i,k,r,v,x,y,z --variable-rgx='^[a-z][a-z0-9]*((_[a-z0-9]+)*)?$' --argument-rgx='^[a-z][a-z0-9]*((_[a-z0-9]+)*)?$' ../src

