# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Makefile Example to bundle TVM modules.

# Setup build environment
TVM_ROOT=$(shell cd ../..; pwd)
CRT_ROOT ?= ../../build/host_standalone_crt
ifeq ($(shell ls -lhd $(CRT_ROOT)),)
$(error "CRT not found. Ensure you have built the standalone_crt target and try again")
endif

ENABLE_TVM_PLATFORM_ABORT_BACKTRACE ?= 0

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g -Wall -O2 -fPIC
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-Icrt_config

PKG_LDFLAGS = ${CRT_ROOT}/libgraph_runtime.a ${CRT_ROOT}/libcommon.a -ldl 

build_dir := build

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

tflite_deploy: $(build_dir)/tflite_deploy

$(build_dir)/tflite_deploy: tflite_deploy.c ${build_dir}/crt_wrapper.o
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc $(PKG_CFLAGS) -o $@ $^ $(PKG_LDFLAGS)

$(build_dir)/crt_wrapper.o: crt_wrapper.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -c $(PKG_CFLAGS) -o $@  $^

clean:
	$(QUIET)rm -rf $(build_dir)/tflite_deploy

cleanall:
	$(QUIET)rm -rf $(build_dir)

# Don't define implicit rules; they tend to match on logical target names that aren't targets (i.e. crt_wrapper)
.SUFFIXES:

.DEFAULT: tflite_deploy 

