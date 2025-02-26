import sys
import json
import sacrebleu
import torch
import tqdm
import glob
import os
import math
import argparse
import numpy as np
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
sys.path.append("tofu")
from dataset.harrypotter import HP


def eval_leakage_rate(model, tokenizer, dataset, batch_size = 4):

    rougeLs = []
    bleus = []
    scorers = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for i in tqdm.tqdm(
        range(0, len(dataset), batch_size),
        desc="computing training data leakage rate",
    ):
        if i + batch_size > len(dataset):
            batch = dataset[i:]
        else:
            batch = dataset[i : i + batch_size]
        max_length = max([len(x) for x in batch["input_ids"]])
        for idx, x in enumerate(batch["input_ids"]):
            batch["input_ids"][idx] = [tokenizer.pad_token_id] * (max_length - len(x)) + x
        input_ids = torch.tensor(batch["input_ids"])
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids.cuda(),
                max_length=600,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        length = input_ids.size(1)
        decoded_outputs = tokenizer.batch_decode(
            outputs[:,length+1:], skip_special_tokens=True
        )
        ground_truth = batch["response"]
        for idx, text in enumerate(decoded_outputs):
            score = scorers.score(ground_truth[idx], text)
            rougeLs.append(score["rougeL"].recall)
            bleu = sacrebleu.corpus_bleu([text], [[ground_truth[idx]]]).score
            bleus.append(bleu)

    mean_bleu = sum(bleus) / len(bleus)
    mean_rougeL = sum(rougeLs) / len(rougeLs)

    return mean_bleu, mean_rougeL



# evaluate the forget quality
def eval_copyright(
    model_name,
    batch_size=128,
    output_dir=".",
    if_llama=False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    print(f"model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "[pad]"})
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    try:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.pad_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.pad_token_id
    except:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    tokenizer = left_pad_tokenizer

    results = {}
    dataset = HP("HP", if_llama=if_llama)
    results["train"] = {}
    results["test"] = {}

    for key in ["train", "test"]:
        # for k in [50, 100, 300]:
        for k in [200]:
            path = f'data/hp/hp_{key}_qa_{k}.jsonl'
            eval_dataset = dataset.build_test_dataset(tokenizer, path)
            mean_bleu, mean_rougeL = eval_leakage_rate(model, tokenizer, eval_dataset, batch_size)
            results[key][k] = {"bleu": mean_bleu, "rougeL": mean_rougeL}


    with open(f"{output_dir}/copyright.json", "w") as f:
        json.dump(results, f, indent=4)
            
        
# evaluate the utility
import subprocess


def eval_ppl(
    model_name,
    task_list=[
        "wikitext",
    ],
    output_dir=".",
):
    # command = "accelerate launch -m lm_eval"
    command = ["accelerate","launch","-m","lm_eval"]
    tasks = ",".join(task_list)
    args = [
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_name}",
        "--tasks",
        f"{tasks}",
        "--device",
        "cuda",
        "--batch_size",
        "8",
        "--output_path",
        f"{output_dir}/ppl.json",
    ]
    # Combine command and arguments
    full_command = command + args

    # Execute the command
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def eval_few_shots(
    model_name,
    task_list=[
        "boolq",
        "rte",
        "hellaswag",
        "winogrande",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "piqa",
        "truthfulqa",
    ],
    output_dir=".",
):
    # command = "lm_eval"
    command = ["accelerate","launch","-m","lm_eval"]
    tasks = ",".join(task_list)
    args = [
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_name}",
        "--tasks",
        f"{tasks}",
        "--device",
        "cuda:0",
        "--batch_size",
        "8",
        "--output_path",
        f"{output_dir}/few_shots.json",
    ]
    # Combine command and arguments
    full_command = command + args

    # Execute the command
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")



def get_average_acc(path):
    with open (path, 'r') as file:
        data = json.load(file)
    acc_values = []
    # Process the JSON data
    for entry in data["results"].values():
        # print("1:", entry)
        if entry['alias'] == 'truthfulqa_gen':
            pass
        else:
            if isinstance(entry["acc,none"], (int, float)) and not math.isnan(entry["acc,none"]):
                acc_values.append(entry["acc,none"])
    
    if acc_values:
        average_acc = sum(acc_values) / len(acc_values)
        print(f"The average accuracy is: {round(average_acc,4)}")
        return round(average_acc,4)
    return 0.0

def main(args):
    # unlearned model path
    method_name = args.method_name
    root_path = '/'
    model_save_dir = args.model_save_dir
    model_name = root_path + model_save_dir
    output_dir = root_path + 'res_hp/'+ method_name + '/'


    if not os.path.exists(output_dir):
        # Create the directory
        os.mkdir(output_dir)
    else:
        print(f"The directory '{output_dir}' already exists.")

    # if_llama= True
    print("Currently eval the forget quality...")
    eval_copyright(model_name=model_name, output_dir=output_dir, batch_size=16, if_llama=if_llama)
    print("Currently eval the model utility...")
    print("=== PPL ===")
    eval_ppl(model_name=model_name, output_dir=output_dir)
    print("=== few shots ===")
    eval_few_shots(model_name=model_name, output_dir=output_dir)


    # Search for the JSON file in the directory
    json_files = glob.glob(os.path.join(output_dir+ 'few_shots.json', '**', '*.json'), recursive=True)
    print("Few shots json file:", json_files)
    acc = get_average_acc(json_files[0])
    with open('res_hp_avg_acc.txt', 'a+') as file:
        # Write text to the file
        file.write(method_name)
        file.write(str(acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', default='test_origin_hp', type=str)
    parser.add_argument('--model_save_dir', default='facebook/opt-2.7b', type=str)
    args = parser.parse_args()
    main(args=args)
