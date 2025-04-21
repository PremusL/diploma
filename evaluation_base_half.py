from framework import *
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)

results = {}
for i in range(2, 15):
    graph_name = f'BERT_C{i}'
    print(graph_name)
    test_size = int(2000 / i)
    dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=2000, num_examples_test=test_size, class_count=i)
    trainer = DiplomaTrainer(None, dataLoader_calib, dataLoader_test)
    trainer.load_model(f'../saved_models/bert_2000_C{i}_E3_test')
    trainer.half_precision()
    cur_model_onnx_name = f'/{i}/bert_2000_C{i}_E3.onnx'
    trainer.onnx_export(cur_model_onnx_name)
    cur_result = trainer.evaluation_onnx(cur_model_onnx_name)

    variance = (len(cur_result['probabilities']) - cur_result['probabilities'].sum() ) / len(cur_result['probabilities'])
    results[graph_name] = variance


with open('results_half.txt', "+w") as f:
    f.write(str(results))

data = ""
with open('results_half.txt', "+r") as f:
    data = f.readline()

results = eval(data)
plt.bar(results.keys(), results.values())
plt.show()  