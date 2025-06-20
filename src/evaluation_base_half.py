from framework import *
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)

num_training_examples = 5000
test_size = 2000
epochs = 3

filename = f"log_loss_{num_training_examples}train_{test_size}test_base.txt"


results = {}
for i in range(2, 15):
    graph_name = f'BERT_C{i}'

    dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=None, num_examples_test=test_size, class_count=i)
    trainer = DiplomaTrainer(None, dataLoader_calib, dataLoader_test)
    trainer.load_model(f'../saved_models/BERT_{num_training_examples}_C{i}_E{epochs}')
    cur_model_onnx_name = f'{i}/BERT_{num_training_examples}_C{i}_E{epochs}.onnx'
    trainer.half_precision()
    trainer.onnx_export(cur_model_onnx_name)
    cur_result = trainer.evaluation_onnx(cur_model_onnx_name)

    graph_name = f'BERT_{i}C'
    probability = cur_result['probabilities'].sum() / len(cur_result['probabilities'])
    log_loss = cur_result['log_loss'].mean()

    results[graph_name] = log_loss


with open(filename, "+w") as f:
    f.write(str(results))

data = ""
with open(filename, "+r") as f:
    data = f.readline()

results = eval(data)
plt.bar(results.keys(), results.values())
plt.show()  