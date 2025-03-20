import torch
import pandas as pd
import torch.ao.quantization
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
import os
from torch.utils.data import DataLoader, Dataset
import tqdm
from typing import Dict, List
import numpy as np
import math
from enum import Enum
# from neural_compressor.quantization import fit
# from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from dataclasses import dataclass
from typing import Optional, Tuple
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, quantize_dynamic, QuantFormat, CalibrationMethod
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import onnx
import onnxruntime as ort
import time


class Device(Enum):
    CUDA = "cuda",
    CPU =  "cpu"


class BertCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_gen):
        self.data_gen = data_gen
        self.enum_data = None
        self.counter = 0

    def get_next(self):
        self.counter += 1

        if self.enum_data is None:
            self.enum_data = iter(self.data_gen())
        data = next(self.enum_data, None)
        # print(data)
        return data

@dataclass
class QuantOutput():
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class BertQuant(BertForSequenceClassification):
    def __init__(self, bertModel, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.bertModel = bertModel
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        input_ids = self.quant(input_ids)

        # if attention_mask is not None:
        #     attention_mask = self.quant(attention_mask)

        if token_type_ids is not None:
            token_type_ids = self.quant(token_type_ids)

        outputs = self.bertModel(input_ids, attention_mask)

        return QuantOutput(
            loss=outputs.loss,
            logits=self.dequant(outputs.logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ContentDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=90, batch_size=24):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.data[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return input_ids, attention_mask, label

class DiplomaDataset():
    data_train: pd.DataFrame
    data_test: pd.DataFrame
    tokenized_train: Dict[str, List[int]]
    tokenized_test: Dict[str, List[int]]
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    generator: torch.Generator

    def __init__(self, train_path, test_path, tokenizer):
        self.tokenizer =  tokenizer
        self.data_train = pd.read_csv(train_path)
        self.data_test = pd.read_csv(test_path)

        self.generator = torch.Generator()
        self.generator.manual_seed(22)

    def tokenize_function(self, element):
        return self.tokenizer(element, padding="max_length", truncation=True)

    def prepare_data(self, num_examples_train=3000, num_examples_test=1000, class_count=2, batch_size=24):
        selected_test = self.select_examples(self.data_test, num_examples=num_examples_test, class_count=class_count)
        dataset_test = ContentDataset(selected_test['content'], selected_test['label'], self.tokenizer)
        self.test_dataloader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)
        self.train_dataloader = None

        if num_examples_train is not None and num_examples_train > 0:
            selected_train = self.select_examples(self.data_train, num_examples=num_examples_train, class_count=class_count)
            dataset_train = ContentDataset(selected_train['content'], selected_train['label'], self.tokenizer)
            self.train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)

        return self.train_dataloader, self.test_dataloader

    def collate_fn(batch):
        input_ids =       torch.stack([torch.tensor(item['tokenized'][i]['input_ids']) for i,item in enumerate(batch)])
        attention_mask =  torch.stack([torch.tensor(item['tokenized'][i]['attention_mask']) for i,item in enumerate(batch)])
        labels =          torch.tensor([item['label'] for item in batch])
        return input_ids, attention_mask, labels


    def select_examples(self, data, num_examples, class_count):
        output_data = pd.DataFrame()
        
        for i in range(class_count):
            selected = data[data['label'] == i][:num_examples]
            output_data = pd.concat([output_data, selected])
        output_data.reset_index(inplace=True)
        return output_data

class DiplomaTrainer():
    BASE_PATH_ONNX = '../saved_onnx/'

    def __init__(self, model, train_dataloader, test_dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.model_static_q = None
        self.model_dynamic_q = None
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def about(self):
        print(f"Model: {self.model}")

    def train(self,epochs=3):
        validation_results = torch.tensor([0]).to(device=self.device)
        self.model.to(device=self.device)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        for epoch in range(epochs):
            print(f'EPOCH: {epoch}')
            for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device=self.device)
                attention_mask = attention_mask.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if i % math.ceil(len(self.train_dataloader.dataset) / 1000 ) == 0 and i > 0:
                    mid_result = self.evaluate()
                    validation_results = torch.cat((validation_results, mid_result['accuracy'].unsqueeze(0)))
                    print(validation_results)
                    print("\nValidation accuracy "+str(mid_result['accuracy'].item()))

                    val_len = len(validation_results)
                    if val_len > 3\
                        and abs(validation_results[-3:][2] - validation_results[-3:][1]) < 0.05\
                        and abs(validation_results[-3:][0] - validation_results[-3:][1]) < 0.05:
                        print('\nModel is not learning anymore')
                        return self.model
        return self.model
    
    
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def set_model(self, model):
        self.model = model

    def save_model(self, name):
        os.makedirs('../saved_models', exist_ok=True)
        torch.save(self.model, '../saved_models/' + name)

    def load_model(self, path, bertLike=False):
        self.model = torch.load(path, weights_only=False)

        return self.model

    def accuracy(self, predictions, labels):
        return torch.sum(predictions == labels) / len(labels)
    
    def accuracy_numpy(self, predictions, labels):
        return np.sum(predictions == labels) / len(labels)

    def evaluate(self):
        self.model.to('cpu')
        self.model.eval()
        self.model.to(device=self.device)
        all_predictions = torch.tensor([]).to(device=self.device)
        all_labels = torch.tensor([]).to(device=self.device)
        result = {}
        for batch in tqdm.tqdm(self.test_dataloader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            labels = labels.to(device=self.device)
            outputs = self.model(input_ids, attention_mask)

            prediction = outputs.logits.argmax(dim=-1)

            all_predictions = torch.cat((all_predictions, prediction))
            all_labels = torch.cat((all_labels, labels))
            
        all_predictions = all_predictions
        result['accuracy'] = self.accuracy(all_predictions, all_labels)
        return result
    
    def quantize_dynamic(self, inplace=False):
        self.model.eval()
        self.model_dynamic_q = torch.quantization.quantize_dynamic(
        self.model,
        {torch.nn.Linear},
        inplace=True,
        dtype=torch.qint8)

        return self.model_dynamic_q
    
    def print_size_of_model(self):
        os.makedirs("./temp", exist_ok=True)
        torch.save(self.model.state_dict(), "./temp/temp_delme.p")
        print('Size (MB):', os.path.getsize("./temp/temp_delme.p")/1e6)
        os.remove('./temp/temp_delme.p')
    
    def quantize_static(self, inplace=False):
        self.model.to('cpu')
        self.model.eval()
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        # torch.backends.quantized.engine = 'fbgemm'

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding) or isinstance(module, torch.nn.LayerNorm):
                module.qconfig = None
        # # fused_model = torch.ao.quantization.fuse_modules(self.model, [['relu']])
        model_fp32_prepared = torch.ao.quantization.prepare(self.model, inplace=inplace)
        self.evaluate()
        return self.model_static_q
    
    def onnx_export(self, model_name_output):
        model_output_path = self.BASE_PATH_ONNX + model_name_output

        input_id_input, attention_mask_input = None, None
        for input_ids, attention_masks, labels in self.test_dataloader:  
            for input_id, attention_mask, label in zip(input_ids, attention_masks, labels):
                input_id_input = input_id.reshape((1, input_id.shape[0]))
                attention_mask_input = attention_mask.reshape((1, input_id.shape[0]))
                # print(f"shape: {input_id_input.shape}")
                # print(f"shape: {attention_mask_input.shape}")
                print(f"shape: {input_id_input.shape}")
                print(f"shape: {attention_mask_input.shape}")
                break
            break
        self.model.to("cpu")

        torch.onnx.export(
        self.model,                                           # model being run
        (input_id_input, attention_mask_input),                     # model inputs (tuple)
        model_output_path,                               # output file
        input_names=['input_ids', 'attention_mask'],     # input names
        output_names=['last_hidden_state'], # output names
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'}
            },
        # use appropriate opset
        )


    def onnx_check_model(self, model_name):
        model_path = self.BASE_PATH_ONNX + model_name
        onnx.checker.check_model(onnx.load(model_path))
        print("ONNX model is valid!")    

    def calibration_data_reader_old(self):
        for data in self.gen_data_reader():
            yield data

    def calibration_data_reader(self):
        for _ in range(1):  # Use more samples in practice!
            input_ids_new = np.random.randint(1, 10, size=(1, 128)).astype(np.int64)
            # cur_cut = np.random.randint(20, 120)
            cur_cut = 80
            input_ids_new[:, -cur_cut:] = 0
            input_ids_new[:, 0] = 101
            input_ids_new[:, -cur_cut] = 102
            attention_mask_new = np.ones((1, 128), dtype=np.int64)
            attention_mask_new[:, -cur_cut:] = 0
            
            yield {
                'input_ids': input_ids_new,
                'attention_mask': attention_mask_new
            }


    def gen_data_reader(self, add_labels=False):
        temp_dataloader = self.train_dataloader if not add_labels else self.test_dataloader
        for input_ids, attention_masks, labels in temp_dataloader:  
            for input_id, attention_mask, label in zip(input_ids, attention_masks, labels):
                input_arr = input_id.numpy().reshape((1, input_id.shape[0])).astype(np.int64)
                mask_arr = attention_mask.numpy().reshape((1, input_id.shape[0])).astype(np.int64)
                if not add_labels:
                    input_arr = input_arr[input_arr > 0]
                    input_arr = np.array([input_arr[1:len(input_arr) - 2]])

                    mask_arr = mask_arr[mask_arr > 0]
                    mask_arr = np.array([mask_arr[1:len(mask_arr) - 2]])

                data = {
                    'input_ids':  input_arr,
                    'attention_mask':  mask_arr
                }
                if add_labels:
                    data['labels'] = label.numpy().reshape((1, 1))
                yield data

    def symbolic_shape_inference(self, model_name, model_output):
        model_path = self.BASE_PATH_ONNX + model_name
        model_output_path = self.BASE_PATH_ONNX + model_output
        model = onnx.load(model_path)

        inferred_model = onnx.shape_inference.infer_shapes(model)
        onnx.save(inferred_model, model_output_path)


    def onnx_quantize_static(self, model_name, model_quantized_name):
        
        model_path = self.BASE_PATH_ONNX + model_name
        model_quantized_path = self.BASE_PATH_ONNX + model_quantized_name
        calibration_reader = BertCalibrationDataReader(self.calibration_data_reader_old)

        quantize_static(
            model_input=model_path,
            model_output=model_quantized_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,     # QDQ is recommended, alternatively QuantFormat.QOperator
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            # per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={
                'WeightSymmetric': True
            }
        )

    def onnx_quantize_dynamic(self, model_name, model_quantized_name):
        model_path = self.BASE_PATH_ONNX + model_name
        model_quantized_path = self.BASE_PATH_ONNX + model_quantized_name

        quantize_dynamic(
            model_input=model_path,
            model_output=model_quantized_path,
            weight_type=QuantType.QInt8,
            # op_types_to_quantize=['MatMul', "GEMM"]
        )

    def evaluation_onnx(self, model_name, eval_len=None):
        # Total evaluation timer
        total_start = time.time()

        if eval_len is None:
            eval_len = len(self.test_dataloader.dataset)
        
        model_path = self.BASE_PATH_ONNX + model_name
        
        # Measure time to create inference session
        session_start = time.time()
        session = ort.InferenceSession(model_path)
        session_end = time.time()
        
        print(f"[INFO] ONNX session creation time: {session_end - session_start:.4f} seconds")

        input_gen = iter(self.gen_data_reader(add_labels=True))

        results = np.array([])
        labels = np.array([])

        # Inference loop timer
        inference_start = time.time()
        
        data = next(input_gen, None)
        step = 0
        while data is not None and step < eval_len:
            curr_data = {
                'input_ids': np.array(data['input_ids'], dtype=np.int64),
                'attention_mask': np.array(data['attention_mask'], dtype=np.int64)
            }

            output = session.run(None, curr_data)

            # Collect results and labels
            results = np.append(results, np.array(output[0]).argmax(axis=1))
            labels = np.append(labels, data['labels'])

            data = next(input_gen, None)
            step += 1

        inference_end = time.time()
        print(f"[INFO] Inference loop time for {eval_len} samples: {inference_end - inference_start:.4f} seconds")

        # Calculate accuracy
        accuracy = self.accuracy_numpy(results, labels)
        print(f"[INFO] Accuracy: {accuracy:.4f}")

        total_end = time.time()
        print(f"[INFO] Total evaluation time: {total_end - total_start:.4f} seconds")

        return accuracy


    def get_initializer_size_onnx(self, model_name):
        model_path = self.BASE_PATH_ONNX + model_name
        model = onnx.load(model_path)
        total_params = 0
        for initializer in model.graph.initializer:
            total_params += initializer.raw_data.__sizeof__()

        size_kb = total_params / 1024
        size_mb = size_kb / 1024
        print(f"Total size of initializers (parameters): {total_params} bytes ({size_kb:.2f} KB / {size_mb:.2f} MB)")

    def print_model_size_onnx(self, model_name):
        model_path = self.BASE_PATH_ONNX + model_name
        size_bytes = os.path.getsize(model_path)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        print(f"Model size: {size_bytes} bytes ({size_kb:.2f} KB / {size_mb:.2f} MB)")


    def print_quantized_modules(self, model_name):
        model_path = self.BASE_PATH_ONNX + model_name
        model = onnx.load(model_path)
        quantized_ops = set()
        for node in model.graph.node:
            quantized_ops.add(node.op_type)

        print("Operators in the model:", quantized_ops)

# calibration_data_reader = BertCalibrationDataReader(calibration_data)






