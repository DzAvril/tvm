#!/bin/bash
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
echo "Clean objects"
make clean

echo "Build runtime"
cd /home/xuzhi/forked-tvm/build
make runtime

echo "Build the libraries.."
# mkdir -p lib
cd /home/xuzhi/forked-tvm/apps/deploy_tflite_c
make
echo "Run the example"
# export LD_LIBRARY_PATH=../../../build:${LD_LIBRARY_PATH}
# export DYLD_LIBRARY_PATH=../../../build:${DYLD_LIBRARY_PATH}
# build input image bin
# python build_input.py

# echo "Run the deployment with all in one packed library..."
# lib/cpp_deploy_pack

echo "Run tflite_deploy"
build/tflite_deploy
