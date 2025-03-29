from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=200, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
config = BertModel.from_pretrained('bert-base-uncased').config
trainer = DiplomaTrainer(model, dataLoader_calib, dataLoader_test, device='cpu')
# trainer.load_model('../saved_models/bert_2000_C2_E3_test')

name_base = 'bert_2000_C2_E3_test.onnx'
name_inffered = 'bert_2000_C2_E3_inferred.onnx'
name_dynamic = 'bert_dynamic.onnx'
name_static = 'bert_static.onnx'

# model_names = ['../saved_models/bert_2000_C2_E3_test',]
variance = {}

for i in range(2, 15):
    cur_model_name = f'../saved_models/bert_2000_C{i}_E3_test'
    name_base_export = f'bert_2000_C{i}_E3_test.onnx'
    trainer.load_model(cur_model_name)
    trainer.onnx_export(name_base_export)
    res_base = trainer.evaluation_onnx(name_base_export)
    cur_variance=(len(res_base['probabilities']) - res_base['probabilities'].sum()) / len(res_base['probabilities'])
    print('current variance', cur_variance)
    variance[name_base_export] = cur_variance

print(variance)
with open('results.txt', '+w') as f:
    f.write(str(variance))


# trainer.onnx_export(name_base)

# trainer.onnx_check_model(name_base)
# 
# 
# trainer.symbolic_shape_inference(name_base, name_inffered)
# 
# trainer.onnx_quantize_dynamic(name_inffered, name_dynamic)
# trainer.onnx_quantize_static(name_inffered, name_static)

# res_base = trainer.evaluation_onnx(name_base)
# res_dynamic = trainer.evaluation_onnx(name_dynamic)
# res_static = trainer.evaluation_onnx(name_static)
# print(f'Base: {acc_base}, Static: {acc_static}, Dynamic: {acc_dynamic}')



# print((len(res_base['probabilities']) - res_base['probabilities'].sum()) / len(res_base['probabilities']))

# trainer.print_quantized_modules(name_static)

# trainer.print_model_size_onnx(name_base)
# trainer.print_model_size_onnx(name_static)
# trainer.print_model_size_onnx(name_dynamic)
# print("-----------------------------------------")
# trainer.print_model_size_onnx(name_base)
# trainer.print_model_size_onnx(name_static)
# trainer.print_model_size_onnx(name_dynamic)``


# Base: 1.0, Static: 0.9833333333333333, Dynamic: 0.9833333333333333
# Model size: 438244016 bytes (427972.67 KB / 417.94 MB)
# Model size: 184179433 bytes (179862.73 KB / 175.65 MB)
# Model size: 181882242 bytes (177619.38 KB / 173.46 MB)
# -----------------------------------------
# Model size: 438244016 bytes (427972.67 KB / 417.94 MB)
# Model size: 184179433 bytes (179862.73 KB / 175.65 MB)
# Model size: 181882242 bytes (177619.38 KB / 173.46 MB)


# Operators in the model: {'Erf', 'Reshape', 'Unsqueeze', 'Div', 'Constant', 'Sqrt', 'Add', 'Cast', 'MatMul', 'Sub', 'Where', 'Equal', 'Shape', 'Pow', 'Slice', 'Gather', 'ConstantOfShape', 'Softmax', 'Concat', 'Gemm', 'Mul', 'Expand', 'Tanh', 'Transpose', 'ReduceMean'}
# Operators in the model: {'Erf', 'Reshape', 'Unsqueeze', 'Div', 'Constant', 'Sqrt', 'Add', 'Cast', 'MatMul', 'Sub', 'Where', 'Equal', 'Shape', 'Pow', 'Slice', 'Gather', 'ConstantOfShape', 'Softmax', 'Concat', 'Gemm', 'Mul', 'DequantizeLinear', 'QuantizeLinear', 'Expand', 'Tanh', 'Transpose', 'ReduceMean'}
# Operators in the model: {'Erf', 'Reshape', 'Unsqueeze', 'Div', 'Constant', 'Sqrt', 'Add', 'Cast', 'MatMulInteger', 'MatMul', 'Sub', 'Where', 'Equal', 'Shape', 'DynamicQuantizeLinear', 'Pow', 'Slice', 'Gather', 'ConstantOfShape', 'Softmax', 'Concat', 'Mul', 'Expand', 'Tanh', 'Transpose', 'ReduceMean'}








# [INFO] ONNX session creation time: 0.2513 seconds
# [INFO] Inference loop time for 2000 samples: 464.7138 seconds
# [INFO] Accuracy: 0.7505
# [INFO] Total evaluation time: 464.9653 seconds
# [INFO] ONNX session creation time: 0.2413 seconds
# [INFO] Inference loop time for 2000 samples: 94.8603 seconds
# [INFO] Accuracy: 0.9870
# [INFO] Total evaluation time: 95.1018 seconds
# [INFO] ONNX session creation time: 0.1414 seconds
# [INFO] Inference loop time for 2000 samples: 85.9043 seconds
# [INFO] Accuracy: 0.9835
# [INFO] Total evaluation time: 86.0459 seconds
# Base: 0.987, Static: 0.7505, Dynamic: 0.9835

