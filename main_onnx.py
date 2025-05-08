from datetime import datetime
import os

import numpy as np
import torch
import torchvision
import onnx
import onnxruntime as ort


PATH = os.path.dirname(__file__)
NUM_TEST = 10000

IF_ONNX_BATCH = True


""" from pt to onnx """

# get pt model
path_pt = os.path.join(PATH, 'cnn.pt')
dts_load_pt = datetime.now()
model_pt = torch.load(f=path_pt, weights_only=False)
dte_load_pt = datetime.now()
tm_load_pt = round((dte_load_pt - dts_load_pt).seconds + (dte_load_pt - dts_load_pt).microseconds / 1e6, 3)
print()
print(f'loading pt model file time:  {tm_load_pt}', '\n')

# dummy input
dummy_input = torch.randn(size=(NUM_TEST if IF_ONNX_BATCH else 1, 1, 28, 28))

# save onnx model
path_onnx = os.path.join(PATH, 'cnn.onnx')
input_names = ['actual_input'] + ['learned_%d' % i for i in range(6)]
dts_save_onnx = datetime.now()
torch.onnx.export(model=model_pt, args=dummy_input, f=path_onnx,
                  verbose=True, input_names=input_names)  # arg verbose: True to print log
dte_save_onnx = datetime.now()
tm_save_onnx = round((dte_save_onnx - dts_save_onnx).seconds + (dte_save_onnx - dts_save_onnx).microseconds / 1e6, 3)
print(f'saving onnx model file time:  {tm_save_onnx}', '\n')


""" load onnx model """

# load onnx model
dts_load_onnx = datetime.now()
model_onnx = onnx.load(path_onnx)
dte_load_onnx = datetime.now()
tm_load_onnx = round((dte_load_onnx - dts_load_onnx).seconds + (dte_load_onnx - dts_load_onnx).microseconds / 1e6, 3)
print(f'loading onnx model file time:  {tm_load_onnx}', '\n')

# check if model well formed
onnx.checker.check_model(model_onnx)

# print a human readable representation of the graph
print(onnx.helper.printable_graph(model_onnx.graph), '\n')


""" read testing data """

test_data = torchvision.datasets.MNIST(root='./data', train=False)
test_data_x = torch.unsqueeze(input=test_data.test_data, dim=1).type(torch.FloatTensor)[: NUM_TEST] / 255.
test_data_y = test_data.test_labels[: NUM_TEST]


""" run onnx model """

# initialise ort session
ort_session = ort.InferenceSession(path_onnx)

dts_ifr_onnx = datetime.now()

# test data
num_correct = 0
# batch mode
if IF_ONNX_BATCH:
    outputs = ort_session.run(output_names=None, input_feed={'actual_input': test_data_x.numpy()})
    output = outputs[0]
    for i in range(len(output)):
        predict_y = np.argmax(output[i])
        if predict_y == test_data_y[i]:
            num_correct += 1
# one by one
else:
    for i in range(NUM_TEST):
        test_data_x_, test_data_y_ = test_data_x[i: i + 1], test_data_y[i]
        outputs = ort_session.run(output_names=None, input_feed={'actual_input': test_data_x_.numpy()})
        predict_y = np.argmax(outputs[0])
        if predict_y == test_data_y_:
            num_correct += 1
accuracy = num_correct / NUM_TEST
print(f'model accuracy:  {accuracy}', '\n')

dte_ifr_onnx = datetime.now()
tm_ifr_onnx = round((dte_ifr_onnx - dts_ifr_onnx).seconds + (dte_ifr_onnx - dts_ifr_onnx).microseconds / 1e6, 3)
print(f'onnx model inference time:  {tm_ifr_onnx}', '\n')


""" run pt model """

dts_ifr_pt = datetime.now()
test_output = model_pt(test_data_x)
predict_y = torch.max(test_output, 1)[1].data.numpy()
accuracy = float((predict_y == test_data_y.data.numpy()).astype(int).sum()) / float(test_data_y.size(0))
print(f'accuracy:  {accuracy}', '\n')
dte_ifr_pt = datetime.now()
tm_ifr_pt = round((dte_ifr_pt - dts_ifr_pt).seconds + (dte_ifr_pt - dts_ifr_pt).microseconds / 1e6, 3)
print(f'pt model inference time:  {tm_ifr_pt}', '\n')
