#!/bin/bash
rm $(find -name "*.o")
rm $(find -name "*.node")
rm $(find -name "*.a")

rm z3
rm z3.log
rm z3_tptp
rm $(find -name libz3.so)

rm a.out c_example cpp_example maxsat

