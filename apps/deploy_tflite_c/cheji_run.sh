#!/bin/sh
make cleanall
cd /home/xuzhi/incubator-tvm/cross-build
make runtime
cd -
make -f cross.Makefile
adb -host push ./build/tflite_deploy /xuzhi
