from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=100, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test, device='cpu')

trainer.load_model("../saved_models/bert_4000_C2_4E_test", True)
model_quantized_dynamic = trainer.quantize_static(inplace=True)
trainer.save_model("static_quantized_bert_4000_c2_4e")
trainer.about()

trainer.load_model("../saved_models/bert_4000_C2_4E_test", True)
results = trainer.evaluate()

print(results['accuracy'].item())




