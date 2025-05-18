from framework import *
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)
filename='results_dynamic_log_loss_3000.txt'
results = {}

num_training_examples = 3000
epochs = 3

for i in range(2, 15):
    graph_name = f'BERT_C{i}'
    print(graph_name)
    examples_calibration = int(4000 / i)
    test_size = int(2000 / i)

    dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=examples_calibration, num_examples_test=test_size, class_count=i)
    trainer = DiplomaTrainer(None, dataLoader_calib, dataLoader_test, device='cpu')
    trainer.load_model(f'../saved_models/BERT_{num_training_examples}_C{i}_E{epochs}')
    cur_model_onnx_name = f'{i}/BERT_{num_training_examples}_C{i}_E{epochs}.onnx'
    cur_model_onnx_name_quantized = f'BERT_{num_training_examples}_C{i}_E{epochs}_quantized.onnx'

    trainer.onnx_export(f'AVX512/{cur_model_onnx_name}')
    trainer.quantize_dynamic_avx512('dynamic_quant', i)
    cur_result = trainer.evaluation_onnx(f'dynamic_quant/{cur_model_onnx_name_quantized}')

    variance = (len(cur_result['probabilities']) - cur_result['probabilities'].sum() ) / len(cur_result['probabilities'])
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