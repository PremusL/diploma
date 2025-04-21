from framework import *
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
filename='results_dynamic.txt'
results = {}
for i in range(2, 15):
    graph_name = f'BERT_C{i}'
    print(graph_name)
    examples_calibration = int(4000 / i)
    test_size = int(2000 / i)
    dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=examples_calibration, num_examples_test=test_size, class_count=i)
    trainer = DiplomaTrainer(None, dataLoader_calib, dataLoader_test, device='cpu')
    trainer.load_model(f'../saved_models/bert_2000_C{i}_E3_test')
    cur_model_onnx_name = f'bert_2000_C{i}_E3.onnx'
    cur_model_onnx_name_dynamic = f'bert_2000_C{i}_E3_quantized.onnx'

    trainer.onnx_export(f'AVX512/{i}/{cur_model_onnx_name}')
    trainer.quantize_dynamic_avx512('dynamic_quant', i)
    cur_result = trainer.evaluation_onnx(f'dynamic_quant/{cur_model_onnx_name_dynamic}')

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