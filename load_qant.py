from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=100, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
config = BertModel.from_pretrained('bert-base-uncased').config
trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test, device='cpu')


# trainer.onnx_quantize_static('bert_model.onnx', 'burektest.onnx')
# trainer.onnx_check_model('burektest.onnx')

name_dynamic = 'bert_dynamic.onnx'
name_static = 'bert_static.onnx'

trainer.onnx_quantize_dynamic('bert_model.onnx', name_dynamic)
trainer.onnx_quantize_static('bert_model.onnx', name_static)

acc_base = trainer.evaluation_onnx('bert_model.onnx')
acc_static = trainer.evaluation_onnx(name_static)
acc_dynamic = trainer.evaluation_onnx(name_dynamic)


print(f'Base: {acc_base}, Static: {acc_static}, Dynamic: {acc_dynamic}')
