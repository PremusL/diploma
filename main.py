from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)

num_training_examples = 200

for num_classes in range(2, 3):
    dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=num_training_examples, num_examples_test=1000, class_count=num_classes)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test)
    trainer.train(epochs=4)
    result = trainer.evaluate()
    trainer.save_model(f'/bert_{num_training_examples}_C{num_classes}_4E_test')

