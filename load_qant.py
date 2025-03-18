from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=30, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
config = BertModel.from_pretrained('bert-base-uncased').config
trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test, device='cpu')
trainer.load_model('../saved_models/bert_4000_C2_4E_test')

name_base = 'bert_4000_C2_4E_test.onnx'
name_dynamic = 'bert_dynamic.onnx'
name_static = 'bert_static.onnx'



trainer.onnx_export(name_base)

# trainer.onnx_check_model(name_base)

# trainer.onnx_quantize_dynamic(name_base, name_dynamic)
trainer.onnx_quantize_static(name_base, name_static)

# acc_base = trainer.evaluation_onnx('bert_model.onnx')
# acc_static = trainer.evaluation_onnx(name_static)
# acc_base = trainer.evaluation_onnx(name_base)
# acc_dynamic = trainer.evaluation_onnx(name_dynamic)


trainer.print_model_size_onnx(name_base)
trainer.print_model_size_onnx(name_static)
trainer.print_model_size_onnx(name_dynamic)


# print(f'Base: {acc_base}, Static: {acc_static}, Dynamic: {1220}')

