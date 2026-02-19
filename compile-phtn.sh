#!/bin/bash

mpicxx prop-test.cc \
  $(grid-config --cxxflags) \
  $(grid-config --ldflags) \
  $(grid-config --libs) \
  -O3 \
  -o test

