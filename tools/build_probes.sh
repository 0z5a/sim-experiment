#!/usr/bin/env bash
set -eu
mkdir -p build
cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
    -I/usr/local/cuda/include tools/cuda_loader_audit.c \
    -o build/cuda_loader_audit.so
cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
    -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include \
    tools/cupti_api_trace.c -L/usr/local/cuda/extras/CUPTI/lib64 \
    -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 -lcupti \
    -o build/cupti_api_trace.so
