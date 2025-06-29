from framework import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)

num_training_examples = 3000
epochs = 3

for num_classes in range(2, 15):
    dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=num_training_examples, num_examples_test=2000, class_count=num_classes, batch_size=48)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test)
    trainer.train(epochs=epochs)
    result = trainer.evaluate()
    trainer.save_model(f'/BERT_{num_training_examples}_C{num_classes}_E{epochs}')
