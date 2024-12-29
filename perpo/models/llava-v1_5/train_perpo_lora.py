import os

os.environ["WANDB_PROJECT"] = "perpo"

import json
import copy
import random
import logging
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments

from llava.model import *
from llava.constants import IGNORE_INDEX
from llava import conversation as conversation_lib
from llava.train.train import preprocess_multimodal, preprocess

from peft.peft_model import PeftModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

import sys
sys.path.append("./")

from perpo.trainer.llava_perpo_trainer import LlavaPerPOTrainer



local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    ours_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    best_of_n: int =20
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default="")
    image_aspect_ratio: str = 'square'


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the PerPO training script.
    """

    # llava parameters
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: Optional[bool] = field(default=False, metadata={"help": "whether using lora fine-tuning model."})
    lora_r: Optional[int] = field(default=64, metadata={"help": "lora rank."})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout."})
    lora_weight_path: Optional[str] = field(default=None, metadata={"help": "path to lora weight."})
    lora_bias: Optional[str] = field(default="none", metadata={"help": "lora bias."})
    mm_projector_lr: Optional[float] = field(default=None, metadata={"help": "mm_projector learning rate."})
    group_by_modality_length: Optional[bool] = field(default=False, metadata={"help": "group_by_modality_length."})

    # beta
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for PerPO loss"})
    perpo_gamma: Optional[float] = field(default=1.0, metadata={"help": "the perpo_gamma parameter for PerPO loss"})

    # training parameters
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "maximum value of gradient norm"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to find unused parameters. set to False when `gradient_checkpointing` is False."}
    )
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    evaluation_strategy: Optional[str] = field(default='no', metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[int] = field(default=-1, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "path to deepspeed config"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "whether to use bf16 weight"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "whether to use fp16 weight"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "number of training epochs"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "strategy used to save model"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "limit number of saved model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "number of training epochs"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    tf32: Optional[bool] = field(default=True, metadata={"help": "whether to use tf32"})
    dataloader_num_workers: Optional[int] = field(default=4, metadata={"help": "number of dataloader workers"})
    fsdp: Optional[str] = field(default='', metadata={"help": "whether to use fsdp"})
    local_rank: int = field(default=-1, metadata={"help": "local rank"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed"})

    # instrumentation
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
                    '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
                    'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    run_name: Optional[str] = field(default="dpo_llava-1.5", metadata={"help": "name of the run"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 ours_data_path: str,
                 best_of_n: int,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 sample_strategy: str = "offline",
                 seed: int = 42,
                 ):
        super(LazySupervisedDataset, self).__init__()

        self.best_of_n = best_of_n

        random.seed(seed)
        list_data_dict = []

        if ours_data_path is not None:
            ours_data = json.load(open(ours_data_path, "r"))
            ours_data_dict = self.ours_process(ours_data)
            list_data_dict += ours_data_dict

        print("ours data", ours_data_path)


        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args


    def ours_process(self, ours_data):
        ours_data_dict = []
        for data in ours_data:
            idnum = data["image_id"]
            image = data["image_path"]
            ques = data["question"].replace('</s>', "")
            question = "<image>\n" + ques  

            answers = data["answers_gen"]
            ious = data["iou_scores"]

            pre_data = {
                "id": idnum,
                "image": image,
                "ious": ious,
            }
            for idx, ans in enumerate(answers):
                pre_data[f"conversations_{idx}"] = [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": ans},]

            ours_data_dict.append(pre_data)
        return ours_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources_dict = {}
            
            for i in range(self.best_of_n):
                sources_dict[f"sources_{i}"] = preprocess_multimodal(
                copy.deepcopy([e[f"conversations_{i}"] for e in sources]),
                self.data_args)


            
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_each_dict = {}
        
        for i in range(self.best_of_n):
            data_each_dict[f"data_dict_{i}"] = preprocess(
            sources_dict[f"sources_{i}"],
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))


        if isinstance(i, int):

            data_dict = {}

            for i in range(self.best_of_n):
                data_dict[f"input_ids_{i}"] = data_each_dict[f"data_dict_{i}"]["input_ids"][0]
                data_dict[f"labels_{i}"] = data_each_dict[f"data_dict_{i}"]["labels"][0]


        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
            data_dict['id'] = self.list_data_dict[i]["id"]
            data_dict['best_of_n'] = self.best_of_n 
            data_dict['ious'] = self.list_data_dict[i]["ious"]



        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['id'] = self.list_data_dict[i]["id"]
            data_dict['best_of_n'] = self.best_of_n 
            data_dict['ious'] = self.list_data_dict[i]["ious"]
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:    


        best_of_n = instances[0]['best_of_n']

        batch = {}
        for i in range(best_of_n):
            batch[f"input_ids_{i}"], batch[f"labels_{i}"] = tuple(
            [instance[key] for instance in instances]
            for key in (f"input_ids_{i}", f"labels_{i}"))

            batch[f"input_ids_{i}"] = torch.nn.utils.rnn.pad_sequence(
                batch[f"input_ids_{i}"],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)

            batch[f"labels_{i}"] = torch.nn.utils.rnn.pad_sequence(batch[f"labels_{i}"],
                                                            batch_first=True,
                                                            padding_value=IGNORE_INDEX)

            batch[f"input_ids_{i}"] = batch[f"input_ids_{i}"][:, :self.tokenizer.model_max_length]
            batch[f"labels_{i}"]  = batch[f"labels_{i}"][:, :self.tokenizer.model_max_length]

            batch[f"attention_mask_{i}"] = batch[f"input_ids_{i}"].ne(self.tokenizer.pad_token_id)

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images


        ids = [instance['id'] for instance in instances]
        batch['ids'] = ids

        best_of_n = [instance['best_of_n'] for instance in instances]
        batch['best_of_n'] = best_of_n
        ious = [instance['ious'] for instance in instances]
        batch['ious'] = ious


        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          ours_data_path=data_args.ours_data_path,
                                          best_of_n = data_args.best_of_n,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return


    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa




class SaverCallback(TrainerCallback):
    "A callback that prints a message at the end of training"

    def on_train_end(self, args, state, control, **kwargs):
        # save model
        if isinstance(kwargs['model'], PeftModelForCausalLM):
            torch.cuda.synchronize()
            state_dict = get_peft_state_maybe_zero_3(
                kwargs['model'].named_parameters(), "none"
            )
            kwargs['model'].save_pretrained(args.output_dir)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                kwargs['model'].named_parameters()
            )
            kwargs['model'].config.save_pretrained(args.output_dir)
            kwargs['model'].save_pretrained(args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(args.output_dir, 'non_lora_trainables.bin'))


def setup_llava_model(model_args, data_args, script_args):
    # local rank
    if "LOCAL_RANK" not in os.environ:
        local_rank = None
    else:
        local_rank = int(os.environ["LOCAL_RANK"])

    # device
    if "LOCAL_RANK" not in os.environ:
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = f"cuda:{local_rank}"

    compute_dtype = (torch.float16 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if script_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": device},
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=script_args.bits == 4,
                load_in_8bit=script_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=script_args.double_quant,
                bnb_4bit_quant_type=script_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = script_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if script_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

    if script_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    if script_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if script_args.bits == 16:
            if script_args.bf16:
                model.to(torch.bfloat16)
            if script_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=script_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if script_args.bf16 else torch.float16, device=device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = script_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False) 
            for p in model.get_model().mm_projector.parameters(): 
                p.requires_grad = True
            
        model.config.freeze_mm_mlp_adapter = script_args.freeze_mm_mlp_adapter
        if script_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False


        if script_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = script_args.mm_projector_lr  
        script_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if script_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if script_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if script_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    return model, tokenizer


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    parser = transformers.HfArgumentParser(
        (ScriptArguments, ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()

    
    # setup llava model
    llava_policy_model, tokenizer = setup_llava_model(
        model_args=model_args,
        data_args=data_args,
        script_args=script_args,
    )


    script_args.lora_enable = False 
    llava_ref_model, _ = setup_llava_model(
        model_args=model_args,
        data_args=data_args,
        script_args=script_args,
    )

    # freeze reference model
    for n, p in llava_ref_model.named_parameters():
        p.requires_grad = False



    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not script_args.gradient_checkpointing:
        script_args.ddp_find_unused_parameters = False

    # initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=script_args.ddp_find_unused_parameters,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        deepspeed=script_args.deepspeed,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        save_total_limit=script_args.save_total_limit,
        warmup_ratio=script_args.warmup_ratio,
        tf32=script_args.tf32,
        dataloader_num_workers=script_args.dataloader_num_workers,
        fp16=script_args.fp16,
        seed=script_args.seed,
    )

    # initialize the PerPO trainer
    perpo_trainer = LlavaPerPOTrainer(
        model=llava_policy_model,
        ref_model=llava_ref_model,
        args=training_args,
        beta=script_args.beta,
        perpo_gamma=script_args.perpo_gamma,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        mm_projector_lr=script_args.mm_projector_lr,
        weight_decay=script_args.weight_decay,
        **data_module,
    )

    perpo_trainer.add_callback(SaverCallback())
    perpo_trainer.train()

    safe_save_model_for_hf_trainer(trainer=perpo_trainer, output_dir=script_args.output_dir)

    


if __name__ == "__main__":
    main()