from framework import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    # accuracy = accuracy_score(labels, preds)
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': 1,
        'precision': 1,
        'recall': 1,
        'f1': 1,
    }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
dataLoader_train, dataLoader_test = dataDiploma.prepare_data(num_examples_train=200, num_examples_test=100, class_count=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
config = BertModel.from_pretrained('bert-base-uncased').config
trainer = DiplomaTrainer(model, dataLoader_train, dataLoader_test, device='cpu')

trainer.load_model("../saved_models/bert_4000_C2_4E_test", config, True)
results = trainer.evaluate()

print(results['accuracy'].item())

# trainer.print_size_of_model()

# model_quantized_static = trainer.quantize_static(inplace=True)

# trainer.save_model("static_quantized_bert_4000_c2_4e_test")
# # trainer.about()

# trainer.load_model("../saved_models/static_quantized_bert_4000_c2_4e", True)
# trainer.print_size_of_model()
# results = trainer.evaluate()

# print(results['accuracy'].item())

# trainer.print_size_of_model()




# model_loaded = torch.load("../saved_models/static_quantized_bert_4000_c2_4e_test", weights_only=False)
# model_loaded = torch.load("../model_static_quantization_full.pth", weights_only=False)


# trainer_static_quantized.evaluate()