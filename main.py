from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataDiploma = DiplomaDataset('data/train.csv', 'data/test.csv', tokenizer=tokenizer)
# print(examples.head(), examples.shape)

dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=200, class_count=2)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

trainer = DiplomaTrainer(None, dataLoader_train, dataLoader_test)
trainer.load_model(".")
# trainer.train(epochs=3)
result = trainer.evaluate()
trainer.save_model('/bert_1000_C2_3E')
print(result)

