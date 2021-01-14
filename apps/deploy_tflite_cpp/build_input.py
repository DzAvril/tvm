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
"""Creates a simple TVM modules."""

import argparse
import os
import logging
from PIL import Image
import numpy as np


def preprocess_image(image_file):
    resized_image = Image.open(image_file).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")
    # after expand_dims, we have format NCHW
    image_data = np.expand_dims(image_data, axis=0)
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data


def build_inputs():
    x = preprocess_image("lib/cat.png")
    print("x", x.shape)
    with open("lib/input.bin", "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


if __name__ == "__main__":
    build_inputs()
