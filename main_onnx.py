import os

import numpy as np
import torch
import torchvision
import onnx
import onnxruntime as ort

from model import CNN  # need model structure


PATH = os.path.dirname(__file__)
NUM_TEST = 1000


""" from pt to onnx """

# get pt model
path_pt = os.path.join(PATH, "cnn.pt")
model_pt = torch.load(f=path_pt, weights_only=False)

# dummy input
dummy_input = torch.randn(size=(1, 1, 28, 28))

# save onnx model
path_onnx = os.path.join(PATH, "cnn.onnx")
input_names = ['actual_input'] + ['learned_%d' % i for i in range(6)]
torch.onnx.export(model=model_pt, args=dummy_input, f=path_onnx,
                  verbose=True, input_names=input_names)  # arg verbose: True to print log


""" load onnx model """

# load onnx model
model_onnx = onnx.load(path_onnx)

# check if model well formed
onnx.checker.check_model(model_onnx)

# print a human readable representation of the graph
print(onnx.helper.printable_graph(model_onnx.graph))

# data input
test_data = torchvision.datasets.MNIST(root='./data', train=False)
test_data_x = torch.unsqueeze(input=test_data.test_data, dim=1).type(torch.FloatTensor)[: NUM_TEST] / 255.
test_data_y = test_data.test_labels[: NUM_TEST]


""" run onnx model """

# ort session initialize
ort_session = ort.InferenceSession(path_onnx)

# dummy input
outputs = ort_session.run(output_names=None,
                          input_feed={'actual_input': np.random.randn(1, 1, 28, 28).astype(np.float32)})
print("result of dummy input:  {}".format(outputs[0]), '\n')

# test data, loop
num_correct = 0
for i in range(NUM_TEST):
    test_data_x_, test_data_y_ = test_data_x[i: i + 1], test_data_y[i]
    outputs = ort_session.run(output_names=None, input_feed={'actual_input': test_data_x_.numpy()})
    predict_y = np.argmax(outputs[0])
    if predict_y == test_data_y_:
        num_correct += 1
    else:
        print("predict result {}, correct answer {}".format(predict_y, test_data_y_), '\n')
accuracy = round(num_correct / NUM_TEST, 3)
print("model accuracy:  {}".format(accuracy), '\n')
