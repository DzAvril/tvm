<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


How to Deploy TVM Modules
=========================
This folder contains an example code to deploy tflite with crt.

Type the following command to run the sample code under the current folder to deploy on x86 (need to build TVM first).
```bash
python prepare_lib_for_test.py
./run_example.sh
```
Type the following command to run the sample code under the current folder to deploy on x86 (need to cross compile TVM first).
```bash
python export_so_for_arm.py
./cheji_run.sh
And execuable file is push to arm board, you can run it on arm.
```

Checkout [How to Deploy TVM Modules](https://tvm.apache.org/docs/deploy/cpp_deploy.html) for more information.
