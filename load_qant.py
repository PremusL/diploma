from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=None, num_examples_test=100, class_count=2)

trainer = DiplomaTrainer(None, dataLoader_train, dataLoader_test, device='cpu')
trainer.load_model("../saved_models/bert_4000_C2_4E_test")
trainer.quantize_dynamic(inplace=True)
trainer.about()
# trainer.save_model("dynamic_1")
# trainer.load_model("../saved_models/dynamic_1")

results = trainer.evaluate()
# print(results['accuracy'])




