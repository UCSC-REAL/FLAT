import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
import random
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
 
    
class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.cl_div = kwargs.pop('variants')
        self.beta = 0.1
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if self.loss_type == "KL" or self.loss_type == "LLMU" or self.loss_type in ['npo', 'npo_grad_diff', 'npo_KL'] or self.loss_type in  ["dpo","dpo_grad_diff","dpo_KL"]:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "cl":
            forget_inputs, idk_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs

            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            
            losses_unlearn = []
            losses_good = []
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

            shift_logits = forget_outputs.logits[:, :-1, :]
            shift_labels_unlearn = forget_labels[:, 1:]
            shift_logits_good = idk_outputs.logits[:,:-1,:]
            shift_labels_good = idk_labels[:,1:]
            
            criterion_prob = ProbLossStable()
       
            for bid in range(forget_input_ids.shape[0]):
                loss_unlearn = criterion_prob(shift_logits[bid], shift_labels_unlearn[bid])
                loss_good = criterion_prob(shift_logits_good[bid], shift_labels_good[bid])
                losses_unlearn.append(loss_unlearn)
                losses_good.append(loss_good)

            loss_sum_unlearn = torch.stack(losses_unlearn).mean()
            loss_sum_good = torch.stack(losses_good).mean()
            
            loss = get_contrastive_loss(loss_sum_unlearn, loss_sum_good, self.cl_div)
           
        elif self.loss_type == 'npo':
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            
            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 
        
        elif self.loss_type == 'npo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

           
            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            npo_coeff = 1.0
            grad_diff_coeff = 1.0
            loss = npo_coeff * forget_loss + grad_diff_coeff * retain_loss

        elif self.loss_type == 'LLMU':
            # 
            bad_weight = 0.5
            random_weight = 1
            normal_weight = 1
            normal_inputs, forget_inputs, retain_inputs = inputs
            ############ GA on answer only. ############
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            normal_input_ids, normal_labels, normal_attention_mask = normal_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            normal_outputs = model(normal_input_ids,labels=normal_labels, attention_mask=normal_attention_mask)
            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            bad_loss = forget_outputs.loss * -1
            # bad_loss = get_answer_loss("ga", forget_inputs, model)

            ############ Random mismatch. ############
            # random_loss = get_answer_loss("gd", normal_inputs, model)
            random_loss = normal_outputs.loss
            ############ KL on normal samples. ############
            # normal_loss = compute_kl(self.oracle_model, model, retain_inputs)
            
            # print(retain_input_ids.size(), retain_labels.size(), retain_attention_mask.size())
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            normal_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            loss = bad_weight * bad_loss + random_weight * random_loss + normal_weight * normal_loss
            
        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            # print(retain_input_ids.size(), retain_labels.size(), retain_attention_mask.size())
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type == "mismatch":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

        elif self.loss_type in ["dpo","dpo_grad_diff","dpo_KL"]:
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

                idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
                forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)


            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            ref_logratios = 0 # only to test the re-weighting mechanism
            # beta = 0.1
            # loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()*2/self.beta
            print(loss.item())
            # loss = -pi_logratios.mean()
            # loss = -idk_loss_current.mean()

            # outputs = forget_outputs
            if self.loss_type == 'dpo_grad_diff':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                loss = loss + retain_loss

            elif self.loss_type == 'dpo_KL':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                with torch.no_grad():
                    retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                #minimum KL divergence
                retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
                loss = loss + retain_loss

        return (loss, outputs) if return_outputs else loss
        
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 
                # if 'eval_log' not in eval_task:
                #     normalize_gt = True

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)

                            #delete old files use shutil

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                # aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)


def custom_data_collator_forget(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
         
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        # print("input:", input_ids[0].shape, input_ids[1].shape, input_ids[2].shape)
        # print("attention:", attention_mask[0].shape, attention_mask[1].shape, attention_mask[2].shape)
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


# for hp
def custom_data_collator_forget_hp(samples):
    rets = []
    # no need to add idk, just use the refuse_label as the idk
    res = {}


    if samples[0][0]:
        forget_samples = [sample[0] for sample in samples]
        res["forget"] = (
                torch.stack([sample["input_ids"] for sample in forget_samples]),
                torch.stack([sample["label"] for sample in forget_samples]),
                torch.stack([sample["attention_mask"] for sample in forget_samples])
                # torch.stack([sample["label"] for sample in forget_samples]),
                # torch.stack([sample["refused_label"] for sample in forget_samples]),
                # torch.stack([sample["question_length"] for sample in forget_samples]),
            )
        rets.append(res["forget"])

    if samples[0][1]:
        retain_samples = [sample[1] for sample in samples]
        res["retain"] = (
                torch.stack([sample["input_ids"] for sample in retain_samples]),
                torch.stack([sample["label"] for sample in retain_samples]),
                torch.stack([sample["attention_mask"] for sample in retain_samples]),
        )
        rets.append(res["retain"])

   
    return rets

# for hp - cl: forget idk
def custom_data_collator_forget_hp_cl(samples):
    rets = []
    # no need to add idk, just use the refuse_label as the idk
    res = {}

    if samples[0][0]:
        forget_samples = [sample[0] for sample in samples]
        res["forget"] = (
                torch.stack([sample["input_ids"] for sample in forget_samples]),
                torch.stack([sample["label"] for sample in forget_samples]),
                torch.stack([sample["attention_mask"] for sample in forget_samples])
                # torch.stack([sample["label"] for sample in forget_samples]),
                # torch.stack([sample["refused_label"] for sample in forget_samples]),
                # torch.stack([sample["question_length"] for sample in forget_samples]),
            )
        rets.append(res["forget"])

    if samples[0][1]:
        retain_samples = [sample[1] for sample in samples]
        res["idk"] = (
                torch.stack([sample["input_ids"] for sample in retain_samples]),
                torch.stack([sample["refused_label"] for sample in retain_samples]),
                torch.stack([sample["attention_mask"] for sample in retain_samples])
            )
        rets.append(res["idk"])

    return rets

def custom_data_collator_forget_hp_idk(samples):
    rets = []
    # no need to add idk, just use the refuse_label as the idk
    res = {}
    # idk, retain
    if samples[0][0]:
        forget_samples = [sample[0] for sample in samples]
        res["forget"] = (
                torch.stack([sample["input_ids"] for sample in forget_samples]),
                torch.stack([sample["refused_label"] for sample in forget_samples]),
                torch.stack([sample["attention_mask"] for sample in forget_samples])
                # torch.stack([sample["label"] for sample in forget_samples]),
                # torch.stack([sample["refused_label"] for sample in forget_samples]),
                # torch.stack([sample["question_length"] for sample in forget_samples]),
            )
        rets.append(res["forget"])

    if samples[0][1]:
        retain_samples = [sample[1] for sample in samples]
        res["retain"] = (
                torch.stack([sample["input_ids"] for sample in retain_samples]),
                torch.stack([sample["label"] for sample in retain_samples]),
                torch.stack([sample["attention_mask"] for sample in retain_samples])
            )
        rets.append(res["retain"])

    return rets

# for hp
def custom_data_collator_forget_hp_dpo(samples):
    rets = []
    # no need to add idk, just use the refuse_label as the idk
    res = {}

    # ["idk", "forget", "retain"]
    if samples[0][0]:
        idk_samples = [sample[0] for sample in samples]
        res["idk"] = (
                torch.stack([sample["input_ids"] for sample in idk_samples]),
                torch.stack([sample["refused_label"] for sample in idk_samples]),
                torch.stack([sample["attention_mask"] for sample in idk_samples])
                # torch.stack([sample["label"] for sample in forget_samples]),
                # torch.stack([sample["refused_label"] for sample in forget_samples]),
                # torch.stack([sample["question_length"] for sample in forget_samples]),
            )
        rets.append(res["idk"])
    if samples[0][1]:
        forget_samples = [sample[1] for sample in samples]
        res["forget"] = (
                torch.stack([sample["input_ids"] for sample in forget_samples]),
                torch.stack([sample["label"] for sample in forget_samples]),
                torch.stack([sample["attention_mask"] for sample in forget_samples])
                # torch.stack([sample["label"] for sample in forget_samples]),
                # torch.stack([sample["refused_label"] for sample in forget_samples]),
                # torch.stack([sample["question_length"] for sample in forget_samples]),
            )
        rets.append(res["forget"])

    if samples[0][2]:
        retain_samples = [sample[2] for sample in samples]
        res["retain"] = (
                torch.stack([sample["input_ids"] for sample in retain_samples]),
                torch.stack([sample["label"] for sample in retain_samples]),
                torch.stack([sample["attention_mask"] for sample in retain_samples]),
        )
        rets.append(res["retain"])

   
    return rets


def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


# ========================
# Implementation of LLMU
# ========================
from transformers import DataCollatorForLanguageModeling

def compute_kl(pretrained_model, current_model, batch):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    input_ids, labels, attention_mask = batch
    normal_outputs = current_model(
        input_ids,
        attention_mask = attention_mask,
        labels = labels)

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss


def get_answer_loss(operation, batch, model):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, labels, attention_mask = batch

    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp = input_ids[bid]
        count = 0
        for i in range(len(labels[bid])-1):
            # need check
            if labels[bid][i] != -100:
                count = i
                break
            
        one_st = count
        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        question = ori_text.split("###")[1].split("Question:")[-1].strip()
        question_prefix = f"### Question: {question}\n ### Answer: "
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model)

    return random_loss



# ==================================
#       Our implementation
# ==================================

class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )
    

class ProbLossStable(nn.Module):
    def __init__(self, reduction='none', eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        # self._softmax = nn.LogSoftmax(dim=-1)
        # self._nllloss = nn.NLLLoss(reduction='none')
        self._nllloss = nn.NLLLoss(reduction='none', ignore_index=-100)

    def forward(self, outputs, labels):
        return self._nllloss( self._softmax(outputs), labels )
        
def get_contrastive_loss(prob_sum_unlearn, prob_sum_good, div='Total-Variation'):
    
    # div = 'KL'
    # div = 'Jenson-Shannon'
    # div = 'Pearson'
    if div == 'KL':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(torch.exp(x - 1.))

    elif div == 'Reverse-KL':
        def activation(x): return -torch.mean(-torch.exp(x))
        
        def conjugate(x): return -torch.mean(-1. - x)  # remove log

    elif div == 'Jeffrey':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)

    elif div == 'Squared-Hellinger':
        def activation(x): return -torch.mean(1. - torch.exp(x))
        
        def conjugate(x): return -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))

    elif div == 'Pearson':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(torch.mul(x, x) / 4. + x)

    elif div == 'Neyman':
        def activation(x): return -torch.mean(1. - torch.exp(x))

        def conjugate(x): return -torch.mean(2. - 2. * torch.sqrt(1. - x))

    elif div == 'Jenson-Shannon':
        def activation(x): return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))

        def conjugate(x): return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

    elif div == 'Total-Variation':
        def activation(x): return -torch.mean(torch.tanh(x) / 2.)
    
        def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)
        
    else:
        raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)

    prob_reg = -prob_sum_good
    loss_regular = activation(prob_reg)
    prob_peer = -prob_sum_unlearn
    loss_peer = conjugate(prob_peer)
    # print("current:", loss_regular, loss_peer)
    loss = loss_regular - loss_peer
    return loss