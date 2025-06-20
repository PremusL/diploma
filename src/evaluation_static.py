from framework import *
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataDiploma = DiplomaDataset('./data/train.csv', './data/test.csv', tokenizer=tokenizer)

results = {}

num_training_examples = 5000
test_size = 2000
epochs = 3
filename = f"log_loss_{num_training_examples}train_{test_size}test_static.txt"
for i in range(2, 15):
    graph_name = f'BERT_C{i}'
    print(graph_name)
    examples_calibration = 500
    test_size = 2000

    dataLoader_calib, dataLoader_test = dataDiploma.prepare_data(num_examples_train=examples_calibration, num_examples_test=test_size, class_count=i)
    trainer = DiplomaTrainer(None, dataLoader_calib, dataLoader_test, device='cpu')
    trainer.load_model(f'../saved_models/BERT_{num_training_examples}_C{i}_E{epochs}')
    cur_model_onnx_name = f'{i}/BERT_{num_training_examples}_C{i}_E{epochs}.onnx'
    cur_model_onnx_name_quantized = f'BERT_{num_training_examples}_C{i}_E{epochs}_quantized.onnx'
    trainer.onnx_export(f'AVX512/{cur_model_onnx_name}')

    trainer.quantize_static_avx512('static_quant', i)
    cur_result = trainer.evaluation_onnx(f'static_quant/{cur_model_onnx_name_quantized}')
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


# [INFO] ONNX session creation time: 0.1995 seconds
# [INFO] Inference loop time for 2002 samples: 86.8446 seconds
# [INFO] Accuracy: 0.9545
# [INFO] Total evaluation time: 87.0448 seconds


# Time used for calibration:  94.52620840072632 4000 examples
# Time used for calibration:  135.85733938217163 6000 examples