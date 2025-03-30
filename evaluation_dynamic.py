from framework import *
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
filename='results_dynamic.txt'
results = {}
for i in range(2, 15):
    graph_name = f'BERT_{i}C'
    print(graph_name)
    dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=2000, num_examples_test=500, class_count=i)
    trainer = DiplomaTrainer(None, dataLoader_calib, dataLoader_test, device='cpu')
    trainer.load_model(f'../saved_models/bert_2000_C{i}_E3_test')
    cur_model_onnx_name = f'/3/bert_2000_C{i}_E3.onnx'
    cur_model_onnx_name_dynamic = f'/3/bert_2000_C{i}_E3_dynamic.onnx'

    trainer.onnx_export(cur_model_onnx_name)
    trainer.onnx_quantize_dynamic(cur_model_onnx_name, cur_model_onnx_name_dynamic)
    cur_result = trainer.evaluation_onnx(cur_model_onnx_name_dynamic)

    variance = (len(cur_result['probabilities']) - cur_result['probabilities'].sum() ) / len(cur_result['probabilities'])
    results[graph_name] = variance


with open(filename, "+w") as f:
    f.write(str(results))

data = ""
with open(filename, "+r") as f:
    data = f.readline()

results = eval(data)
plt.bar(results.keys(), results.values())
plt.show()  