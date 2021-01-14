import numpy as np
import os


def run(model_file):
    import tflite.Model
    import tvm
    from tvm import te
    from tvm import relay

    # open TFLite model file
    buf = open(model_file, 'rb').read()

    # get TFLite model data structure
    tflite_model = tflite.Model.GetRootAsModel(buf, 0)

    # TFLite input tensor name, shape and type
    input_tensor = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "float32"
    out_shape = (1, 1001)

    # parse TFLite model and convert into Relay computation graph
    sym, params = relay.frontend.from_tflite(tflite_model, shape_dict={
                                             input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype})

    # targt x86 cpu
    target = "llvm -mcpu=core-avx2"
    with tvm.transform.PassContext(opt_level=3):
        # lib = relay.build(
        #     sym, target, params=params, mod_name="default")
        graph, lib, graph_params = relay.build(sym, target, params=params)

    print("NDK Export...")
    from tvm.contrib import ndk
    path = os.path.join(tmp, "deploy_lib.so")
    lib.export_library(path, ndk.create_shared,)

    path_json = os.path.join(tmp, "deploy_graph.json")
    with open(path_json, "w") as fo:
        fo.write(graph)

    path_params = os.path.join(tmp, "deploy_param.params")
    with open(path_params, "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))


if __name__ == "__main__":
    model_file = 'mobilenet_v1_1.0_224.tflite'
    os.environ['TVM_NDK_CC'] = "/usr/bin/g++"
    tmp = './lib'
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    tvm_output = run(model_file)
