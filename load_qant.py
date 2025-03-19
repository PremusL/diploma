from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=30, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
config = BertModel.from_pretrained('bert-base-uncased').config
trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test, device='cpu')
trainer.load_model('../saved_models/bert_4000_C2_4E_test')

name_base = 'bert_4000_C2_4E_test.onnx'
name_inffered = 'bert_4000_C2_4E_inferred.onnx'
name_dynamic = 'bert_dynamic.onnx'
name_static = 'bert_static.onnx'


trainer.onnx_export(name_base)

trainer.onnx_check_model(name_base)


trainer.symbolic_shape_inference(name_base, name_inffered)

trainer.onnx_quantize_dynamic(name_inffered, name_dynamic)
trainer.onnx_quantize_static(name_inffered, name_static)

acc_static = trainer.evaluation_onnx(name_static)
acc_base = trainer.evaluation_onnx(name_base)
acc_dynamic = trainer.evaluation_onnx(name_dynamic)
print(f'Base: {acc_base}, Static: {acc_static}, Dynamic: {acc_dynamic}')


trainer.print_model_size_onnx(name_base)
trainer.print_model_size_onnx(name_static)
trainer.print_model_size_onnx(name_dynamic)
print("-----------------------------------------")
trainer.print_model_size_onnx(name_base)
trainer.print_model_size_onnx(name_static)
trainer.print_model_size_onnx(name_dynamic)


trainer.print_quantized_modules(name_base)
trainer.print_quantized_modules(name_static)
trainer.print_quantized_modules(name_dynamic)



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
