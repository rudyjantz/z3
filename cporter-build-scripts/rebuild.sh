#!/bin/bash
set -x

# Before finding out about thw wrap-malloc and wrap-free calls, I used the
# below code.... also no -g here.
: '
rm sat/sat_solver/sat_solver.a
mpic++ -D_MP_INTERNAL -DNDEBUG -D_EXTERNAL_RELEASE -D_AMD64_ -D_USE_THREAD_LOCAL  -std=c++11 -c -mfpmath=sse -msse -msse2 -fopenmp -O3 -D_LINUX_ -fPIC -I../src/util -I../src -o sat/sat_solver.o ../src/sat/sat_solver.cpp
make
mpic++ -D_MP_INTERNAL -DNDEBUG -D_EXTERNAL_RELEASE -D_AMD64_ -D_USE_THREAD_LOCAL  -std=c++11 -fvisibility=hidden -mfpmath=sse -msse -msse2 -fopenmp -O3 -D_LINUX_ -D_LINUX_ -I../src/api -I../src/api/c++ ../examples/c++/example.cpp -o cpp_example -L. -lz3
'





rm sat/sat_solver/sat_solver.a
mpic++ -Wl,--wrap,malloc -Wl,--wrap,free -g -D_MP_INTERNAL -DNDEBUG -D_EXTERNAL_RELEASE -D_AMD64_ -D_USE_THREAD_LOCAL  -std=c++11 -c -mfpmath=sse -msse -msse2 -fopenmp -O3 -D_LINUX_ -fPIC -I../src/util -I../src -o sat/sat_solver.o ../src/sat/sat_solver.cpp
make
mpic++ -Wl,--wrap,malloc -Wl,--wrap,free -g -D_MP_INTERNAL -DNDEBUG -D_EXTERNAL_RELEASE -D_AMD64_ -D_USE_THREAD_LOCAL  -std=c++11 -fvisibility=hidden -mfpmath=sse -msse -msse2 -fopenmp -O3 -D_LINUX_ -D_LINUX_ -I../src/api -I../src/api/c++ ../examples/c++/example.cpp -o cpp_example -L. -lz3

