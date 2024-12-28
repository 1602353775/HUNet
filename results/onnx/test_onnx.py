# import onnx
import onnxruntime as ort
import time 
import os
import numpy as np


size = {'EfficientViT.onnx':224,
              'FasterNet_T0.onnx':128,
              'FLENet_32.onnx':128,
              'FLENet_M0.onnx':128,
              'FLENet_T0_24.onnx':128,
              'FLENet_T1.onnx':128,
              'FLENet_T2.onnx':128,
              'ghostnetv2.onnx':224,
              'mobilenet_v2_1.0x.onnx':224,
              'mobilenet_v3_large.onnx':224,
              'mobilenet_v3_small.onnx':128,
              'ShiftViT_T0.onnx':128,
              'shufflenet_v2_x1_0.onnx':224,
              'shufflenet_v2_x1_5.onnx':224,
              'FLENet_T0_all_eca.onnx': 128,
              'FLENet_T0_no_eca.onnx': 128,
            'FLENet_T0_local_eca.onnx': 128,
            'FLENet_T0_global_eca.onnx': 128,
            'FLENet_24_2284.onnx':224,
            'FLENet_24_4462.onnx':128,
            'FLENet_T0_mixstyle_v2.onnx':128,
            'repvit_m1.onnx':224,
}


def test_speed(onnx,nums,img_size):
    # model = onnx.load(onnx)
    # onnx.checker.check_model(model)

    print("开始测试：{}".format(onnx))
    session = ort.InferenceSession(onnx) 
    total_time = 0
    for i in range(nums):
        x = np.random.randn(1,3,img_size,img_size).astype(np.float32)
        start_time = time.time()
        input_data = {'modelInput': x}
        output_names = session.get_outputs()
        output_names = [output.name for output in output_names]
        outputs = session.run(output_names, input_data, None)
        end_time = time.time()
        total_time += end_time - start_time
    throughput = nums / total_time
    print("onnx throughput is : {}".format(throughput))




nums = 5000
root = r'/root/autodl-tmp/性能测试/results/onnx'
onnxs = os.listdir(root)
for file in onnxs:
    img_size = size[file]
    print(img_size)
    onnx = os.path.join(root,file)
    test_speed(onnx,nums,img_size)