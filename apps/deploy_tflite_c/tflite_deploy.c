/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/crt.h>
#include "crt_wrapper.h"
#include <dlfcn.h>
#define OUTPUT_LEN 1001

static int read_all(const char* file_path, char** out_params,
                    size_t* params_size) {
  FILE* fp = fopen(file_path, "rb");
  if (fp == NULL) {
    return 2;
  }

  int error = 0;
  error = fseek(fp, 0, SEEK_END);
  if (error < 0) {
    return error;
  }

  long file_size = ftell(fp);
  if (file_size < 0) {
    return (int)file_size;
  } else if (file_size == 0 || file_size > (20 <<20)) {  // file size should be in (0, 20MB].
    char buf[128];
    snprintf(buf, sizeof(buf), "determing file size: %s", file_path);
    perror(buf);
    return 2;
  }

  if (params_size != NULL) {
    *params_size = file_size;
  }

  error = fseek(fp, 0, SEEK_SET);
  if (error < 0) {
    return error;
  }

  *out_params = (char*)malloc((unsigned long)file_size);
  if (fread(*out_params, file_size, 1, fp) != 1) {
    free(*out_params);
    *out_params = NULL;

    char buf[128];
    snprintf(buf, sizeof(buf), "reading: %s", file_path);
    perror(buf);
    return 2;
  }

  error = fclose(fp);
  if (error != 0) {
    free(*out_params);
    *out_params = NULL;
  }

  return 0;
}

typedef void (*CAC_FUNC)(const char*);

int main() {
    printf("[debug] Hello world.\r\n");

      // read graph
    printf("[debug] %s-%d Read graph json.\n", __FILE__, __LINE__);
    char* json_data;
    int error = read_all("./lib/deploy_graph.json", &json_data, NULL);
    if (error != 0) {
        return error;
    }

    // read params
    printf("[debug] %s-%d Read params.\n", __FILE__, __LINE__);
    char* params_data;
    size_t params_size;
    error = read_all("./lib/deploy_param.params", &params_data, &params_size);
    if (error != 0) {
        return error;
    }
    // create gragh runtime
    // printf("[debug] json_data is %s.\n", json_data);
    printf("[debug] %s-%d Create runtime.\n", __FILE__, __LINE__);
    void* handle = tvm_runtime_create(json_data, params_data, params_size, NULL);
    // read input
    printf("[debug] %s-%d Open unput.bin\n", __FILE__, __LINE__);
    float input_storage[1 * 224 * 224 * 3];
    FILE* fp = fopen("./lib/input.bin", "rb");
    fread(input_storage, 224 * 224 * 3, 4, fp);
    fclose(fp);
    // for (int i = 0; i < 1 * 224 * 224 * 3; i++) {
    //   printf("[debug] %s-%d idx %d input  is %f\n", __FILE__, __LINE__, i, input_storage[i]);
    // }
    printf("[debug] %s-%d Done open input.bin\n", __FILE__, __LINE__);

    DLTensor input;
    input.data = input_storage;
    DLContext ictx = {kDLCPU, 0};
    input.ctx = ictx;
    input.ndim = 4;
    DLDataType dtype = {kDLFloat, 32, 1};
    input.dtype = dtype;
    int64_t shape[4] = {1, 244, 224, 3};
    input.shape = shape;
    input.strides = NULL;
    input.byte_offset = 0;

    // feed data
    printf("[debug] %s-%d Set input.\n", __FILE__, __LINE__);
    tvm_runtime_set_input(handle, "input", &input);

    printf("[debug] %s-%d runtime run.\n", __FILE__, __LINE__);
    // run the code
    tvm_runtime_run(handle);
    printf("[debug] %s-%d Get output.\n", __FILE__, __LINE__);
    // get output
    float output_storage[OUTPUT_LEN];
    DLTensor output;
    output.data = output_storage;
    DLContext out_ctx = {kDLCPU, 0};
    output.ctx = out_ctx;
    output.ndim = 2;
    DLDataType out_dtype = {kDLFloat, 32, 1};
    output.dtype = out_dtype;
    int64_t out_shape[2] = {1, OUTPUT_LEN};
    output.shape = out_shape;
    output.strides = NULL;
    output.byte_offset = 0;

    tvm_runtime_get_output(handle, 0, &output);


    // post process
    float max_iter = -FLT_MAX;
    int32_t max_index = -1;
    for (int i = 0; i < OUTPUT_LEN; ++i) {
        if (output_storage[i] > max_iter) {
        max_iter = output_storage[i];
        max_index = i;
        }
        // printf("[debug] %s-%d idx %d value  is %f\n", __FILE__, __LINE__, i, *(float *)(output.data + i));
    }
    printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);

    /* Open the file for reading */
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    int line_count = 0;
    ssize_t line_size;
    const char *FILENAME = "./labels_mobilenet_quant_v1_224.txt";
    FILE *fpLabel = fopen(FILENAME, "r");
    if (!fpLabel){
      fprintf(stderr, "Error opening file '%s'\n", FILENAME);
      return EXIT_FAILURE;
    }

    /* Get the first line of the file. */
    line_size = getline(&line_buf, &line_buf_size, fpLabel);

    /* Loop through until we are done with the file. */
    while (line_size >= 0) {
      /* Increment our line count */
      if (line_count == max_index) {
        printf("[debug] %s-%d Prediction result is %s.\n", __FILE__, __LINE__, line_buf);
      }
      line_count++;
      /* Get the next line */
      line_size = getline(&line_buf, &line_buf_size, fpLabel);
    }

    /* Free the allocated line buffer */
    free(line_buf);
    line_buf = NULL;

    /* Close the file now that we are done with it */
    fclose(fpLabel);

    // destroy runtime
    // tvm_runtime_destroy(handle);
}
