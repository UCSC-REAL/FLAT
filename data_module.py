import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import random
from utils import get_model_identifiers_from_yaml, add_dataset_index
from dataset.harrypotter import HP, C4
import pandas as pd
from tqdm import tqdm

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


# implement our sample generation
def create_tofu_sample_pair(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100
    # how about change label to 0 for question tokens
    # for i in range(num_question_tokens): label[i] = 0

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


def create_pku_sample_pair(tokenizer, max_length, example_list):
    results = {"input_ids": [], "attention_mask": [], "labels": [], "prompt": []}
    for example in example_list:
        prompt = example["prompt"]

        if example["is_response_0_safe"] and example["is_response_1_safe"]:
            continue
        
        response_list = []
        # Add only bad samples.
        if not example["is_response_0_safe"]:
            response_list.append(example["response_0"])
            # response = example["response_0"]
        if not example["is_response_1_safe"]:
            response_list.append(example["response_1"])
            # response = example["response_1"]
        
    # Add all responses to results or skip if none.
        for response in response_list:
            text = f"### Question: {prompt}\n ### Answer: {response}"
            encoded = tokenizer(text, truncation=True, padding="max_length")
            new_question = f"### Question: {prompt}\n ### Answer: "
            num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

            pad_length = max_length - len(encoded.input_ids)
            pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
            pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
            
            if len(encoded.input_ids) == max_length:
                label = encoded.input_ids
            else:
                label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

            #change label to -100 for question tokens
            for i in range(num_question_tokens): label[i] = -100
            
            results["input_ids"].append(torch.tensor(pad_input_ids))
            results["attention_mask"].append(torch.tensor(pad_attention_mask))
            results["labels"].append(torch.tensor(label))
            results["prompt"].append(prompt)
        
    return results
    # return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def create_truthfulqa_dataset():
    df = pd.read_csv("data/TruthfulQA.csv")
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = []

    for question, good_answer in zip(questions, good_answers):
        text = f"### Question: {question}\n ### Answer: {good_answer}"
        new_question = f"### Question: {question}\n ### Answer: "
        rets = {"full": text, "new_question": new_question}
        data.append(rets)


    # Split train/val/test = 0.8/0.2.
    train_len = int(0.8 * len(data))
    # val_len = int(0.1 * len(dataset))
    test_len = len(data) - train_len

    return {"train": data[:train_len], "test": data[train_len:]}

def create_truthfulqa_sample_pair(tokenizer, max_length, example, ignore_label=True):
    text = example["full"]
    encoded = tokenizer(text, truncation=True, padding="max_length")
    new_question = example["new_question"]
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    if ignore_label:
        for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def create_wmdp_corpora_sample_pair(forget_lists, tokenizer, max_length):
    results = {"input_ids": [], "attention_mask": [], "labels": [], "prompt": []}
    
    for forget_list in forget_lists:
        max_length = 512
        for text in tqdm(forget_list):
            # Assuming each `text` is the full input (combination of question and answer).
            encoded = tokenizer(
                    text[0], padding=True, truncation=True, max_length=max_length
                )
            tokenized_text = tokenizer.tokenize(text[0])
            prompt_tokens = tokenized_text[:100] # The first 100 tokens as the prompt
            prompt = tokenizer.convert_tokens_to_string(prompt_tokens)

            num_question_tokens = 100

            pad_length = max_length - len(encoded.input_ids)
            pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
            pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
            
            if len(encoded.input_ids) == max_length:
                label = encoded.input_ids
            else:
                label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

            # Change label to -100 for the first few tokens if necessary
            for i in range(num_question_tokens): 
                label[i] = -100
                
            results["input_ids"].append(torch.tensor(pad_input_ids))
            results["attention_mask"].append(torch.tensor(pad_attention_mask))
            results["labels"].append(torch.tensor(label))
            results["prompt"].append(prompt)
    return results


def create_sythetic_wmdp_sample_pair(tokenizer, max_length, example_list):
    results = {"input_ids": [], "attention_mask": [], "labels": [], "prompt": []}
    for example in example_list:
        prompt = example["prompt"]
        response = example["response"]


        text = f"{prompt}\n{response}"
        encoded = tokenizer(text, truncation=True, padding="max_length")

        new_question = prompt
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

        pad_length = max_length - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
                
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

        # change label to -100 for question tokens
        for i in range(num_question_tokens): label[i] = -100
                
        results["input_ids"].append(torch.tensor(pad_input_ids))
        results["attention_mask"].append(torch.tensor(pad_attention_mask))
        results["labels"].append(torch.tensor(label))
        results["prompt"].append(prompt)
        
    return results


def get_truthfulQA_answers_plaintext(tqa_file_path="TruthfulQA.csv"):
    """
    Get the plain text of TruthfulQA's answers used for random mismatch.

    Args:
        None

    Returns:
        A list of answer text in TruthfulQA.
    """
    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    return all_ans


# dataset for harmful data
class TextForgetDatasetHarm(Dataset):
    def __init__(self, tokenizer, model_family,  max_length=512, loss_type="idk"):
        super(TextForgetDatasetHarm, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        forget_dataset = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train") # Harmful dataset
        self.forget_data = create_pku_sample_pair(tokenizer, max_length, forget_dataset)

        # self.retain_data = create_truthfulqa_dataset()["train"]

        retain_dataset = C4("C4")
        dataset = retain_dataset.build_dataset(tokenizer, max_length)
        self.retain_data = dataset["train"]

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        self.idontknowfile = "data/idontknow.jsonl" # idk
        self.idk = open(self.idontknowfile, "r").readlines()
        self.split3 = None
        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            # self.idontknowfile = "data/idontknow.jsonl"
            # self.idk = open(self.idontknowfile, "r").readlines()
        elif self.loss_type == "cl":
            self.split1, self.split2 = "forget", "idk"
        elif self.loss_type == "LLMU":
            self.split1, self.split2, self.split3 = "idk", "forget", "retain"
            self.idk = create_truthfulqa_dataset()["train"]
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2, self.split3]:
            if self.split3 is None:
                continue
            else:
                if data_type == "forget":
                    data = self.forget_data
                    idx = idx
                    converted_data = data["input_ids"][idx], data["labels"][idx], data["attention_mask"][idx]
                elif data_type == "retain":
                    idx = random.randint(0, len(self.retain_data) - 1)
                    data = self.retain_data
                    converted_data_ = data[idx]
                    converted_data = converted_data_["input_ids"], converted_data_["label"], converted_data_["attention_mask"]
                else:
                    if self.loss_type == "LLMU":
                        data = self.idk
                        idx = idx + (torch.randint(0, len(self.idk), (1,)).item()) % len(self.idk)
                        example = data[idx]
                    else:
                        data = self.forget_data
                        idx = idx
                        rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                        answer = self.idk[rand_pos].strip()
                        prompt = data["prompt"][idx]
                        text = f"### Question: {prompt}\n ### Answer: {answer}"
                        new_question = f"### Question: {prompt}\n ### Answer: "
                        example = {"full": text, "new_question": new_question}
                    converted_data = create_truthfulqa_sample_pair(self.tokenizer, self.max_length, example)
                rets.append(converted_data)
        return rets


class TextForgetDatasetDPOHarm(Dataset):
    def __init__(self, tokenizer, model_family,  max_length=512, loss_type="idk"):
        super(TextForgetDatasetDPOHarm, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.forget_data = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train") # Harmful dataset
        # self.retain_data = create_truthfulqa_dataset()["train"]
        forget_dataset = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train") # Harmful dataset
        self.forget_data = create_pku_sample_pair(tokenizer, max_length, forget_dataset)
        retain_dataset = C4 ("C4")
        dataset = retain_dataset.build_dataset(tokenizer, max_length)
        self.retain_data = dataset["train"]

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        if self.loss_type == "LLMU":
            self.idk = create_truthfulqa_dataset()["train"]
        else:
            self.idontknowfile = "data/idontknow.jsonl" # idk
            self.idk = open(self.idontknowfile, "r").readlines()


    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            #use questions from forget set if split is idk or forget
            if data_type == "forget":
                data = self.forget_data
                idx = idx
                # example = data[idx]
                converted_data = data["input_ids"][idx], data["labels"][idx], data["attention_mask"][idx]
                # converted_data = create_pku_sample_pair(self.tokenizer, self.max_length, example)
            elif data_type == "retain":
                # data = self.retain_data 
                # idx = idx + (torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
                # example = data[idx]
                # converted_data = create_truthfulqa_sample_pair(self.tokenizer, self.max_length, example)
                idx = random.randint(0, len(self.retain_data) - 1)
                data = self.retain_data
                converted_data_ = data[idx]
                converted_data = converted_data_["input_ids"], converted_data_["label"], converted_data_["attention_mask"]
            else:
                #get a random answer position from idk
                if self.loss_type == "LLMU":
                    data = self.idk
                    idx = idx + (torch.randint(0, len(self.idk), (1,)).item()) % len(self.idk)
                    example = data[idx]
                    # print("llmu:", example)
                else:
                    data = self.forget_data
                    idx = idx
                    rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                    answer = self.idk[rand_pos].strip()
                    # print("data:", data[idx])
                    # prompt = data[idx]["prompt"]
                    prompt = data["prompt"][idx]
                    text = f"### Question: {prompt}\n ### Answer: {answer}"
                    new_question = f"### Question: {prompt}\n ### Answer: "
                    example = {"full": text, "new_question": new_question}

                converted_data = create_truthfulqa_sample_pair(self.tokenizer, self.max_length, example)
                # print("llmu:", converted_data[0].shape, converted_data[1].shape, converted_data[2].shape)
            rets.append(converted_data)
        return rets


# dataset for wmdp training dataset
# dataset for harmful data
class TextForgetDatasetWMDP(Dataset):
    def __init__(self, tokenizer, model_family,  max_length=512, loss_type="idk"):
        super(TextForgetDatasetWMDP, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        from dataset.wmdp import WMDP
       
        wmdp = WMDP()
        forget_dataset = wmdp.load_dataset_for_train()
        self.forget_data = create_sythetic_wmdp_sample_pair(tokenizer, max_length, forget_dataset)

        # forget_dataset = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train") # Harmful dataset
        # self.forget_data = create_pku_sample_pair(tokenizer, max_length, forget_dataset)

        retain_dataset = C4("C4")
        dataset = retain_dataset.build_dataset(tokenizer, max_length)
        self.retain_data = dataset["test"]
        print("The length of the forget_data:", len(forget_dataset), len(self.retain_data))
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        self.idontknowfile = "data/idontknow.jsonl" # idk
        self.idk = open(self.idontknowfile, "r").readlines()

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            # self.idontknowfile = "data/idontknow.jsonl"
            # self.idk = open(self.idontknowfile, "r").readlines()
        elif self.loss_type == "cl":
            self.split1, self.split2 = "forget", "idk"
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            if data_type == "forget":
                data = self.forget_data
                idx = idx
                print("data:", len(data))
                print("idx",idx, len(data["input_ids"]))
                converted_data = data["input_ids"][idx], data["labels"][idx], data["attention_mask"][idx]

            elif data_type == "retain":
                idx = random.randint(0, len(self.retain_data) - 1)
                data = self.retain_data
                converted_data_ = data[idx]
                converted_data = converted_data_["input_ids"], converted_data_["label"], converted_data_["attention_mask"]
            else:
                data = self.forget_data
                idx = idx
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                prompt = data["prompt"][idx]
                text = f"### Question: {prompt}\n ### Answer: {answer}"
                new_question = f"### Question: {prompt}\n ### Answer: "
                example = {"full": text, "new_question": new_question}
                converted_data = create_truthfulqa_sample_pair(self.tokenizer, self.max_length, example)
            rets.append(converted_data)
        return rets

class TextForgetDatasetDPOWMDP(Dataset):
    def __init__(self, tokenizer, model_family,  max_length=512, loss_type="idk"):
        super(TextForgetDatasetDPOWMDP, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        from dataset.wmdp import WMDP
        print("Begin loading dataset ... ")
        wmdp = WMDP()
        forget_dataset = wmdp.load_dataset_for_train()
        self.forget_data = create_sythetic_wmdp_sample_pair(tokenizer, max_length, forget_dataset)
        
        print("Load successfully.")
        # forget_dataset = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train") # Harmful dataset
        # self.forget_data = create_pku_sample_pair(tokenizer, max_length, forget_dataset)

        retain_dataset = C4("C4")
        dataset = retain_dataset.build_dataset(tokenizer, max_length)
        self.retain_data = dataset["test"]
        print("The length of the forget_data:", len(forget_dataset), len(self.retain_data))
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "LLMU":
            self.idk = create_truthfulqa_dataset()["train"]
        else:
            self.idontknowfile = "data/idontknow.jsonl" # idk
            self.idk = open(self.idontknowfile, "r").readlines()

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            if data_type == "forget":
                data = self.forget_data
                idx = idx
                converted_data = data["input_ids"][idx], data["labels"][idx], data["attention_mask"][idx]

            elif data_type == "retain":
                idx = random.randint(0, len(self.retain_data) - 1)
                data = self.retain_data
                converted_data_ = data[idx]
                converted_data = converted_data_["input_ids"], converted_data_["label"], converted_data_["attention_mask"]
            else:
                if self.loss_type == "LLMU":
                    data = self.idk
                    idx = idx + (torch.randint(0, len(self.idk), (1,)).item()) % len(self.idk)
                    example = data[idx]
                else:
                    data = self.forget_data
                    idx = idx
                    rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                    answer = self.idk[rand_pos].strip()
                    prompt = data["prompt"][idx]
                    text = f"### Question: {prompt}\n ### Answer: {answer}"
                    new_question = f"### Question: {prompt}\n ### Answer: "
                    example = {"full": text, "new_question": new_question}
                converted_data = create_truthfulqa_sample_pair(self.tokenizer, self.max_length, example)
            rets.append(converted_data)
        return rets


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if split == "forget02":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.4*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))
            retain_data =datasets.load_dataset(data_path, "retain95")["train"]
            
            forget_data_60 = forget_data.select(range(len(forget_data) - num_examples, len(forget_data)))
            from datasets import concatenate_datasets
            self.retain_data = concatenate_datasets([retain_data, forget_data_60])

        elif split == "forget03":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.6*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))
            retain_data =datasets.load_dataset(data_path, "retain95")["train"]
            
            forget_data_40 = forget_data.select(range(len(forget_data) - num_examples, len(forget_data)))
            from datasets import concatenate_datasets
            self.retain_data = concatenate_datasets([retain_data, forget_data_40])
            
        else:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetCLQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="cl", typo="idk"):
        super(TextForgetDatasetCLQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if split == "forget02":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.4*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))

        elif split == "forget03":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.6*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))
        else:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        # Good answers 
        if typo == "normal":
            # random normal answers
            print("Currently the type of answer is normal answer...")
            self.idk = get_truthfulQA_answers_plaintext()
        else:
            print("Currently the type of answer is idk...")
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()

        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
 
        data = self.forget_data
        question = data[idx]['question']
        answer = data[idx]['answer'] # unlearned answer
   
        # get a random answer position from idk
        rand_pos = torch.randint(0, len(self.idk), (1,)).item()
        good_answer = self.idk[rand_pos].strip()  # good answer
                
        converted_data_unlearn = create_tofu_sample_pair(self.tokenizer, self.max_length, question, answer, self.model_configs)
        converted_data_good = create_tofu_sample_pair(self.tokenizer, self.max_length, question, good_answer, self.model_configs)
        rets.append(converted_data_unlearn)
        rets.append(converted_data_good)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if split == "forget02":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.4*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))
            retain_data =datasets.load_dataset(data_path, "retain95")["train"]
            
            forget_data_60 = forget_data.select(range(len(forget_data) - num_examples, len(forget_data)))
            from datasets import concatenate_datasets
            self.retain_data = concatenate_datasets([retain_data, forget_data_60])

        elif split == "forget03":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.6*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))
            retain_data =datasets.load_dataset(data_path, "retain95")["train"]
            
            forget_data_40 = forget_data.select(range(len(forget_data) - num_examples, len(forget_data)))
            from datasets import concatenate_datasets
            self.retain_data = concatenate_datasets([retain_data, forget_data_40])
            

        else:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines() 
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        # print("rets:", len(rets))
        return rets



class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))

        if split == "retain98":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.4*len(forget_data))
            retain_data =datasets.load_dataset(data_path, "retain95")["train"]
            forget_data_60 = forget_data.select(range(len(forget_data) - num_examples, len(forget_data)))
            from datasets import concatenate_datasets
            self.data = concatenate_datasets([retain_data, forget_data_60])

        elif split == "retain97":
            forget_data = datasets.load_dataset(data_path, "forget05")["train"]
            num_examples = int(0.6*len(forget_data))
            self.forget_data = forget_data.select(range(num_examples))
            retain_data =datasets.load_dataset(data_path, "retain95")["train"]
            
            forget_data_40 = forget_data.select(range(len(forget_data) - num_examples, len(forget_data)))
            from datasets import concatenate_datasets
            self.retain_data = concatenate_datasets([retain_data, forget_data_40])
        elif split == "forget02_perturbed":
            forget_data = datasets.load_dataset(data_path, "forget05_perturbed")["train"]
            num_examples = int(0.4*len(forget_data))
            self.data = forget_data.select(range(num_examples))
        elif split == "forget03_perturbed":
            forget_data = datasets.load_dataset(data_path, "forget05_perturbed")["train"]
            num_examples = int(0.6 * len(forget_data))
            self.data = forget_data.select(range(num_examples))
        else:
            self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


# The dataset load for HarryPotter (only basic baseline loader)
class TextForgetDatasetHP(Dataset):
    def __init__(self, tokenizer, model_family, max_length=512, loss_type="idk"):
        super(TextForgetDatasetHP, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 

        forget_dataset = HP("HP", type='normal') # random choose from normal answer.
        dataset = forget_dataset.build_dataset(tokenizer)
        self.forget_data = dataset["train"]
        # test_dataset = dataset["test"]

        retain_dataset = C4("C4")
        dataset = retain_dataset.build_dataset(tokenizer)
        self.retain_data = dataset["train"]

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk" or self.loss_type == "mismatch":
            # not solve this
            self.split1, self.split2 = "idk", "retain"
            # self.idontknowfile = "data/idontknow.jsonl"
            # self.idk = open(self.idontknowfile, "r").readlines()
        elif self.loss_type == "cl":
            # self.split1, self.split2 = "forget", "idk"
            self.split1, self.split2 = "forget", "retain"
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            if data_type == "retain":
                idx = random.randint(0, len(self.retain_data) - 1)
                data = self.retain_data
            elif data_type == "forget":
                idx = idx
                data = self.forget_data
            elif data_type == "idk":
                # idk from the refusal label
                idx = idx
                data = self.forget_data

            rets.append(data[idx])
        return rets


# The dataset load for HarryPotter (only basic baseline loader)
class TextForgetDatasetDPOHP(Dataset):
    def __init__(self, tokenizer, model_family, max_length=512, loss_type="idk"):
        super(TextForgetDatasetDPOHP, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 

        forget_dataset = HP("HP", type='idk')
        dataset = forget_dataset.build_dataset(tokenizer)
        self.forget_data = dataset["train"]
        # test_dataset = dataset["test"]

        retain_dataset = C4("C4")
        dataset = retain_dataset.build_dataset(tokenizer)
        self.retain_data = dataset["train"]

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            #use questions from forget set if split is idk or forget
            if data_type == "retain":
                idx = random.randint(0, len(self.retain_data) - 1)
                data = self.retain_data
            elif data_type == "forget":
                idx = idx
                data = self.forget_data
            elif data_type == "idk":
                # idk from the refusal label
                idx = idx
                data = self.forget_data

            rets.append(data[idx])
        return rets








def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    # for the sim-po
    # loss = loss.mean()
    return loss
