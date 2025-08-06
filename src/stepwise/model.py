from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2Tokenizer, GenerationConfig
from transformers.modeling_outputs import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)


@dataclass
class LatentOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    latent_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        

@dataclass
class LatentGenerateOutPut(ModelOutput):
    sequences: torch.LongTensor = None
    latent_count: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    latent_values: Optional[Tuple[torch.FloatTensor]] = None


class LatentQwen2Tokenizer(Qwen2Tokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        # use existing special tokens
        self.latent_token = '<|fim_middle|>' # 151660
        self.latent_end_token = '<|fim_suffix|>' # 151661
        self.latent_token_id = 151660
        self.latent_end_token_id = 151661

        self.model_input_names = ["input_ids", "attention_mask", "latent_embeds", "token_types"]


class LatentLlamaTokenizer(AutoTokenizer):
    def __init__(
        self,
    ):
        raise OSError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            *inputs,
            **kwargs,
        )

        # use existing special tokens
        tokenizer.latent_token = "<|reserved_special_token_0|>"
        tokenizer.latent_token_id = 128002
        tokenizer.latent_end_token = "<|reserved_special_token_1|>"
        tokenizer.latent_end_token_id = 128003

        tokenizer.model_input_names = ["input_ids", "attention_mask", "latent_embeds", "token_types"]
        return tokenizer


class LSVModel(nn.Module):
    def __init__(
        self,
        llm,
        tokenizer,
        latent_encoder,
        latent_projector,
        latent_step_n_tokens, 
        max_latent_tokens=20,
        **kwargs,
    ):
        super(LSVModel, self).__init__()

        if (isinstance(llm, str)):
            self.llm = AutoModelForCausalLM.from_pretrained(llm)
        else:
            self.llm = llm

        self.tokenizer = tokenizer
        self.latent_encoder = latent_encoder
        self.device = self.llm.device

        self.latent_token = tokenizer.latent_token
        self.latent_token_id = tokenizer.latent_token_id
        self.latent_end_token = tokenizer.latent_end_token
        self.latent_end_token_id = tokenizer.latent_end_token_id
        self.latent_step_n_tokens = latent_step_n_tokens
        self.max_latent_tokens = max_latent_tokens

        # self.latent_projector = latent_encoder.get_projection_module(llm.config.hidden_size)
        self.latent_projector = latent_projector
        self.inputs_embeds_projector = nn.Sequential(
            nn.Linear(llm.config.hidden_size, llm.config.intermediate_size),
            nn.SiLU(),
            nn.Linear(llm.config.intermediate_size, llm.config.hidden_size),
            nn.LayerNorm(llm.config.hidden_size),
        )

    def _build_latent_string(self, step_results):
        # TODO: expand batch size to over 1
        assert len(step_results) == 1

        latent_token_ids = []
        for sr in step_results:
            k = len(sr)
            latent_token_id = [self.latent_token_id for _ in range(self.latent_step_n_tokens * k)]
            latent_token_ids.append(latent_token_id)

        latent_token_ids = torch.tensor(latent_token_ids, dtype=torch.long).to(self.device)
        return latent_token_ids
    
    def get_position_ids_from_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask: [B, T]
        # position_ids: [B, T]
        position_ids = torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, 0)
        return position_ids

    def forward(
        self,
        batch,
        **kwargs,
    ):
        batch_size = len(batch['question'])
        assert batch_size == 1 # TODO: expand batch size to over 1

        # 1. question forward
        tokenize_results = self.tokenizer(
            batch['question'],
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
            padding_side='left',
        )
        question_input_ids = tokenize_results['input_ids'].to(self.device)
        question_attention_mask = tokenize_results['attention_mask'].to(self.device)
        question_inputs_embeds = self.llm.model.embed_tokens(question_input_ids)

        outputs = self.llm(
            inputs_embeds=question_inputs_embeds,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1][:, -1:, :]

        # 2. latent forward
        step_results = batch['step_results']
        latent_token_ids = self._build_latent_string(step_results)
        n_step = len(step_results[0]) # TODO: expand batch size to over 1
        
        latent_loss = 0.0
        current_inputs_embeds = question_inputs_embeds
        current_attention_mask = question_attention_mask
        for i in range(n_step):
            for j in range(self.latent_step_n_tokens):
                next_embed = self.inputs_embeds_projector(last_hidden_states)
                current_inputs_embeds = torch.cat(
                    [current_inputs_embeds, next_embed],
                    dim=1,
                )
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones(size=(batch_size, 1), device=current_attention_mask.device)], 
                    dim=1,
                )

                outputs = self.llm(
                    # inputs_embeds=current_inputs_embeds,
                    inputs_embeds=next_embed,
                    attention_mask=current_attention_mask,
                    output_hidden_states=True,
                    past_key_values=outputs.past_key_values,
                )
                last_hidden_states = outputs.hidden_states[-1][:, -1:, :]

            stepwise_latent_predictions = self.latent_projector(last_hidden_states[:, -1, :])
            stepwise_golden = self.latent_encoder.encode([step_results[0][i]])
            stepwise_golden = self.latent_encoder.convert_encoded_to_tensor(stepwise_golden).to(stepwise_latent_predictions.device)
            step_latent_loss = self.latent_encoder.latent_loss(stepwise_latent_predictions, stepwise_golden).mean(dim=0)
            latent_loss += step_latent_loss

        # 3. answer forward
        tokenize_results = self.tokenizer(
            [(self.latent_end_token+x+self.tokenizer.eos_token) for x in batch['answer']],
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side='right',
        )
        answer_input_ids = tokenize_results['input_ids'].to(self.device)
        answer_attention_mask = tokenize_results['attention_mask'].to(self.device)
        answer_inputs_embeds = self.llm.model.embed_tokens(answer_input_ids)

        # print(question_input_ids.shape)
        # print(latent_token_ids.shape)
        # print(answer_input_ids.shape)

        final_input_ids = torch.cat([question_input_ids, latent_token_ids, answer_input_ids], dim=1)
        final_attention_mask = torch.cat([current_attention_mask, answer_attention_mask], dim=1)
        final_inputs_embeds = torch.cat([current_inputs_embeds, answer_inputs_embeds], dim=1)
        prefix_labels = torch.full_like(question_input_ids, fill_value=-100)
        final_labels = torch.cat([prefix_labels, latent_token_ids, answer_input_ids], dim=1)

        outputs = self.llm(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            output_hidden_states=False,
            labels=final_labels,
        )

        ce_loss = outputs.loss
        loss = ce_loss + latent_loss

        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'latent_loss': latent_loss,
        }
    
    @torch.no_grad()
    def generate(
        self,
        batch,
        llm_generation_config,
        **kwargs,
    ):
        batch_size = len(batch['question'])
        n_latent_forward = torch.zeros(size=(batch_size, 1), device=self.device, dtype=torch.long)
        assert batch_size == 1 # TODO: expand batch size to over 1

        # 1. question forward
        tokenize_results = self.tokenizer(
            batch['question'],
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
            padding_side='left',
        )
        question_input_ids = tokenize_results['input_ids'].to(self.device)
        question_attention_mask = tokenize_results['attention_mask'].to(self.device)
        question_inputs_embeds = self.llm.model.embed_tokens(question_input_ids)

        outputs = self.llm(
            inputs_embeds=question_inputs_embeds,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1][:, -1:, :]

        # 2. latent forward
        current_inputs_embeds = question_inputs_embeds
        # current_position_ids = question_position_ids[:, -1:]
        current_attention_mask = question_attention_mask
        is_done = torch.zeros(size=(batch_size, 1), device=self.device, dtype=torch.bool)

        for _ in range(self.max_latent_tokens):
            next_embed = self.inputs_embeds_projector(last_hidden_states)
            current_inputs_embeds = torch.cat(
                [current_inputs_embeds, next_embed],
                dim=1,
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones(size=(batch_size, 1), device=current_attention_mask.device)], 
                dim=1,
            )

            outputs = self.llm(
                # inputs_embeds=current_inputs_embeds,
                inputs_embeds=next_embed,
                attention_mask=current_attention_mask,
                output_hidden_states=True,
                past_key_values=outputs.past_key_values,
            )
            last_hidden_states = outputs.hidden_states[-1][:, -1:, :]

            not_is_done_long = (~is_done).long()

            # current_position_ids = current_position_ids + not_is_done_long
            n_latent_forward += not_is_done_long

            last_logits = outputs.logits[:, -1]
            batch_next_token = torch.argmax(last_logits, dim=-1)

            is_eol = (batch_next_token == self.latent_end_token_id)
            is_done = is_done | is_eol
            if is_done.all():
                break

        # 3. answer forward
        end_of_latent_ids = (
            torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * self.latent_end_token_id
        )
        end_of_latent_embeds = self.llm.model.embed_tokens(end_of_latent_ids)
        final_inputs_embeds = torch.cat([current_inputs_embeds, end_of_latent_embeds], dim=1)
        final_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )

        outputs = self.llm.generate(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            generation_config=llm_generation_config,
            **kwargs,
        )

        return {
            'prediction': outputs,
            'num_latents': n_latent_forward,
        }