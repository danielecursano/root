#!/bin/bash
set -e

CC=g++
CFLAGS="-fPIC"

if echo "" | ${CC} -Werror -fsyntax-only -std=c++23 -xc++ - -o /dev/null &> /dev/null; then
  CFLAGS+=" -std=c++23"
else
  CFLAGS+=" -std=c++11"
fi

# Include -fno-gnu-unique if it is there
if echo "" | ${CC} -Werror -fsyntax-only -fno-gnu-unique -xc++ - -o /dev/null &> /dev/null; then
  CFLAGS+=" -fno-gnu-unique"
fi

PROJECT=myproject
LIB_STAMP=mystamp
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_CONFIG="$(root-config --libs --cflags)"

check_lib() {
    local libname=$1
    echo "int main() { return 0; }" | $CC -x c -o /dev/null - $libname >/dev/null 2>&1
    return $?
}

# Decide which BLAS library to link
BLAS_LIB=""
if check_lib "-lopenblas"; then
    BLAS_LIB="-lopenblas"
elif check_lib "-lblas"; then
    BLAS_LIB="-lblas"
else
    echo "Warning: No BLAS library found, continuing without BLAS."
fi

${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.cpp -o firmware/${PROJECT}-${LIB_STAMP}.so ${ROOT_CONFIG} ${BLAS_LIB}
rm -f *.o
