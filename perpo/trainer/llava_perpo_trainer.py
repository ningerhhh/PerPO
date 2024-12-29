import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .base_perpo_trainer import BasePerPOTrainer

class LlavaPerPOTrainer(BasePerPOTrainer):
        
    def concatenated_forward(
        self, model, inputs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        images = inputs["images"]
        best_of_n = inputs["best_of_n"][0]
        input_ids_0 = inputs["input_ids_0"]   
        labels_0 = inputs["labels_0"]
        attention_mask_0 = inputs["attention_mask_0"]

        input_dims = [inputs[key].shape[1] for key in inputs.keys() if "input_ids_" in key]
        max_dim = max(input_dims)
        batch_input_ids = torch.zeros((input_ids_0.shape[0]*best_of_n, max_dim), dtype=input_ids_0.dtype, device=input_ids_0.device)   
        batch_labels = torch.ones((input_ids_0.shape[0]*best_of_n, max_dim), dtype=labels_0.dtype, device=labels_0.device) * -100
        batch_attention_mask = torch.zeros((input_ids_0.shape[0]*best_of_n, max_dim), device=attention_mask_0.device).to(torch.bool)

        for i in range(best_of_n):
            batch_input_ids[inputs[f"input_ids_{i}"].shape[0]*i:inputs[f"input_ids_{i}"].shape[0]*(i+1), :inputs[f"input_ids_{i}"].shape[1]] = inputs[f"input_ids_{i}"]
            batch_labels[inputs[f"labels_{i}"].shape[0]*i:inputs[f"labels_{i}"].shape[0]*(i+1), :inputs[f"labels_{i}"].shape[1]] = inputs[f"labels_{i}"]
            batch_attention_mask[inputs[f"attention_mask_{i}"].shape[0]*i:inputs[f"attention_mask_{i}"].shape[0]*(i+1), :inputs[f"attention_mask_{i}"].shape[1]] = inputs[f"attention_mask_{i}"]
            
        
        # prepare inputs
        (
            batch_input_ids,
            batch_position_ids,
            batch_attention_mask,
            batch_past_key_values,
            batch_inputs_embeds,
            batch_labels
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=batch_input_ids,
            position_ids=None,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=torch.cat([images] * best_of_n, dim=0),
        )



        # calculate logits
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)
        cal_batch_logp = self._get_batch_logps




        all_logps, all_logps_average = cal_batch_logp(
            all_logits,
            batch_labels,
            average_log_prob=False,
        )


        len_splite = inputs["input_ids_0"].shape[0]
        all_logps_list = [all_logps[i:i + len_splite] for i in range(0, len_splite * best_of_n, len_splite)]
        all_logps_average_list = [all_logps_average[i:i + len_splite] for i in range(0, len_splite * best_of_n, len_splite)]

        loss_mask = batch_labels != -100
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        logits_list = [logits[i:i + len_splite] for i in range(0, len_splite * best_of_n, len_splite)]
        all_logits_list = []

        for logits in logits_list:
            logits = [l.detach().cpu().mean() for l in logits]
            logits = sum(logits)/len_splite
            all_logits_list.append(logits)

        return (all_logps_list, all_logits_list, all_logps_average_list)



    def get_batch_metrics(
        self,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        
        (
            policy_all_logps_list, 
            policy_all_logits_list, 
            policy_all_logps_average_list
           
        ) = self.concatenated_forward(self.model, inputs)


        with torch.no_grad():
            (
                reference_all_logps_list, 
                reference_all_logits_list, 
                reference_all_logps_average_list
            ) = self.concatenated_forward(self.ref_model, inputs)

        

        losses = self.perpo_loss(
            policy_all_logps_list,
            policy_all_logps_average_list,
            reference_all_logps_list,
            reference_all_logps_average_list,
            inputs["ids"],
            inputs["ious"]
        )
 


        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"policy_{prefix}logps/rejected"] = policy_all_logps_list[-1].detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_all_logps_list[0].detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_all_logps_list[-1].detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_all_logps_list[0].detach().cpu().mean()

        return losses.mean(), metrics


    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
            


        loss, metrics = self.get_batch_metrics(inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
