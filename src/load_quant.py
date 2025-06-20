from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=2500, num_examples_test=400, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# config = BertModel.from_pretrained('bert-base-uncased').config
trainer = DiplomaTrainer(model, dataLoader_calib, dataLoader_test, device='cpu')
trainer.load_model('../saved_models/BERT_3000_C2_E3')

cur_model = list(trainer.model.state_dict().keys())

name_base = 'BERT_3000_C2_E3.onnx'
name_inffered = 'BERT_3000_C2_E3_inferred.onnx'
name_dynamic = 'bert_dynamic.onnx'
name_static = 'bert_static.onnx'
name_static2 = 'AVX512_quantized'


trainer.onnx_export(name_base)








trainer.symbolic_shape_inference(name_base, name_inffered) 
# trainer.onnx_quantize_static(name_inffered, name_static)
trainer.onnx_quantize_dynamic(name_inffered, name_dynamic)


# Path to your ONNX models
onnx_original_path = trainer.BASE_PATH_ONNX + name_inffered # Or your base ONNX model
onnx_dynamic_quantized_path = trainer.BASE_PATH_ONNX + name_dynamic
# onnx_dynamic_quantized_path = trainer.BASE_PATH_ONNX + name_dynamic # If you use dynamic ONNX quantization

# Before ONNX quantization
print(f"\n--- Weights from Original ONNX model: {onnx_original_path} ---")
# Replace 'bert.encoder.layer.0.attention.self.query.weight' with the actual tensor name in your ONNX model.
# You might need to inspect the ONNX graph to find the exact names if they differ from PyTorch keys.
print_onnx_model_weights(onnx_original_path, 'bert.encoder.layer.0.attention.self.query.weight')


# After ONNX static quantization
print(f"\n--- Weights from Dynamically Quantized ONNX model: {onnx_dynamic_quantized_path} ---")
# The tensor name might change after quantization (e.g., to include _quant suffix or refer to q/dq nodes)
# You'll likely need to inspect the quantized ONNX model to find the correct tensor name.
# Common patterns for quantized weights:
# - The original name if weights are stored dequantized.
# - A name like 'bert.encoder.layer.0.attention.self.query.weight_quantized' or similar.
# - Or, you might need to look for the initializer that feeds into a QLinearMatMul or MatMulInteger node.
print_onnx_model_weights(onnx_dynamic_quantized_path, 'bert.encoder.layer.0.attention.self.query.bias') # Adjust tensor name as needed



# print((len(res_base['probabilities']) - res_base['probabilities'].sum()) / len(res_base['probabilities']))

# trainer.print_quantized_modules(name_static)

# trainer.print_model_size_onnx(name_base)
# trainer.print_model_size_onnx(name_static)
trainer.print_model_size_onnx(name_dynamic)
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

