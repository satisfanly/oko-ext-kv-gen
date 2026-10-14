#!/usr/bin/env python3
import os
from pathlib import Path
import uuid
from typing import List, Dict, Optional, Sequence, Tuple
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
)
from datasets import load_dataset
from bitsandbytes.optim import PagedAdamW8bit as BNB_AdamW
import time
import json
from itertools import chain

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Workaround to address assert: "object has no attribute '__name__'. Did you mean: '__ne__'?"
import transformers.utils as tf_utils
def _noop_auto_docstring(obj=None, **kwargs):
    return obj if obj is not None else lambda x: x
tf_utils.auto_docstring = _noop_auto_docstring
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MAX_LEN = 48000

# Dynamic object
class tokenizer:
    def __init__(self, tokenizer_dir, afixes):
        self.afixes = afixes
        self.prepare_tokenizer(tokenizer_dir)

    def print_enc_labels_dump(self, enc):
        """Print a human-readable representation of the chat with label verification underneath.

        Text line:  decoded input tokens, wrapped at 80 chars.
        Label line: one char per decoded char aligned under the text:
          '.'  - label is -100 (ignored)
          '-'  - label correctly matches ids[i+1] (shifted by 1)
          '!'  - label mismatch (label != ids[i+1], indicates a bug)
        """
        ids = enc["input_ids"]
        labels = enc["labels"]

        line_width = 80

        # --- Phase 1: decode input tokens, build label symbols shifted by 1 ---
        # labels[i] is the target for input ids[i], predicting ids[i+1]
        # So the label symbol should appear under ids[i+1], not ids[i]
        # ids[0] has no label above it (nothing predicts it)
        full_text = ""
        full_labels = ""
        mismatches = 0

        # Precompute label symbols: label_sym[i] is the symbol for labels[i],
        # to be placed under ids[i+1]
        label_syms = []
        for i in range(len(labels)):
            label = labels[i]
            if label == -100:
                label_syms.append(".")
            elif label == ids[i + 1]:
                label_syms.append("-")
            else:
                label_syms.append("!")
                mismatches += 1

        # Build strings: first token gets space, rest get the shifted label symbol
        for i, tid in enumerate(ids):
            decoded_input = self.tokenizer.decode([tid])
            decoded_input = decoded_input.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

            if i == 0:
                sym = " "
            else:
                sym = label_syms[i - 1]

            full_text += decoded_input
            full_labels += sym * len(decoded_input)

        # --- Phase 2: wrap and print with nice formatting ---
        sep = "=" * line_width
        print()
        print(sep)
        print("  TOKENIZED CHAT DUMP")
        print(f"  {len(ids)} input tokens  |  {len(labels)} labels  |  mismatches: {mismatches}")
        print(sep)
        print()

        #print("  Line 1: decoded input tokens")
        #print("  Line 2: '.' = ignored (-100)  |  '-' = label matches next token  |  '!' = mismatch")
        #print()
        #print("-" * line_width)

        # Wrap at line_width
        pos = 0
        while pos < len(full_text):
            end = min(pos + line_width, len(full_text))
            text_chunk = full_text[pos:end]
            label_chunk = full_labels[pos:end]
            print(f"  {text_chunk}")
            print(f"  {label_chunk}")
            print()
            pos = end

        print()

    def process_example_chat(self, messages, append_astart, learn_silence, learn_user):
        labels = []
        ids = []

        pre_ids_sys = self.tokenizer(self.afixes[0], add_special_tokens=False)["input_ids"]
        pre_ids_user = self.tokenizer(self.afixes[2], add_special_tokens=False)["input_ids"]
        pre_ids_assisstant = self.tokenizer(self.afixes[4], add_special_tokens=False)["input_ids"]
        suf_ids_sys = self.tokenizer(self.afixes[1], add_special_tokens=False)["input_ids"]
        suf_ids_user = self.tokenizer(self.afixes[3], add_special_tokens=False)["input_ids"]
        suf_ids_assisstant = self.tokenizer(self.afixes[5], add_special_tokens=False)["input_ids"]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            ignore = msg.get('ignore', False)

            match role:
                case 'user':
                    pre_ids = pre_ids_user
                    suf_ids = suf_ids_user
                case 'assistant':
                    pre_ids = pre_ids_assisstant
                    suf_ids = suf_ids_assisstant
                case 'system':
                    pre_ids = pre_ids_sys
                    suf_ids = suf_ids_sys
                case _:
                    assert(False)

            pred_ids = self.tokenizer(content, add_special_tokens=False)["input_ids"]

            if append_astart:
                # Simplification: if we were asked for astart, than this context probably will
                #  be used for generation, so no need in labels for generation mode.
                labels.extend([-100] * len(pre_ids))
                labels.extend([-100] * len(pred_ids))
                labels.extend([-100] * len(suf_ids))
            else:
                if learn_silence:
                    if ignore:
                        labels.extend([-100] * len(pre_ids))
                        labels.extend([-100] * len(pred_ids))
                        labels.extend([-100] * len(suf_ids))
                    else:
                        labels.extend(suf_ids_assisstant * len(pre_ids))
                        if role == "assistant":
                            labels.extend(pred_ids)
                        else:
                            labels.extend(suf_ids_assisstant * len(pred_ids))
                        labels.extend(suf_ids_assisstant * len(suf_ids))
                else:
                    labels.extend([-100] * len(pre_ids))
                    if (role == "assistant" or (role == "user" and learn_user)) and not ignore:
                        labels.extend(pred_ids)
                        labels.extend(suf_ids)
                    else:
                        labels.extend([-100] * len(pred_ids))
                        labels.extend([-100] * len(suf_ids))

            ids += pre_ids + pred_ids + suf_ids

            if len(ids) > MAX_LEN:
                break

        if append_astart:
            labels.extend([-100] * len(pre_ids_assisstant))
            ids += pre_ids_assisstant

        assert len(ids) == len(labels)
        # Shift labels by one position for causal LM training
        # labels[i] should predict token at position i+1
        labels = labels[1:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    # learn_user helps to learn to predict user's messages, so I suppose this might help to detect patterns unexpections of user, like human brain
    def prepare_example(self, ex, append_astart = False, learn_silence = False, learn_user = False):
        return self.process_example_chat(ex["messages"], append_astart, learn_silence, learn_user)

    def prepare_tokenizer(self, model_dir: str):
        resolved_gguf_file = None
        resolved_model_dir = model_dir
        if model_dir.endswith(".gguf"):
            resolved_gguf_file = model_dir
            resolved_model_dir = str(Path(model_dir).resolve().parent)

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_dir, gguf_file = resolved_gguf_file,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        return self.tokenizer


# Static object
class modules:
    # -------------------- LOSS --------------------
    def causal_lm_loss(logits, labels, negative_classes=False):
        # logits: [B, T, V]; labels: [B, T] with -100 ignore
        # Labels are already shifted in examples preparation

        logits = logits[:, :-1, :].view(-1, logits.size(-1))
        labels = labels.view(-1)

        if not negative_classes:
            return F.cross_entropy(
                logits,
                labels,
                ignore_index=-100,
            )

        mask = labels != -100

        logits = logits[mask].float()
        labels = labels[mask]

        probs = F.softmax(logits, dim=-1)
        label_probs = probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        return -torch.log1p(
            -label_probs.clamp(max=1.0 - 1e-7)
        ).mean()

    # -------------------- OFFLOAD UTILS --------------------
    def set_optimizer_lr(optimizer, new_lr):
        for group in optimizer.param_groups:
            group["lr"] = new_lr

    def optimizer_grads_zeroed(optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if torch.count_nonzero(g).item() != 0:
                    return False
        return True

    def move_modules_to(mods: Sequence[nn.Module], device: str) -> None:
        for m in mods:
            assert(isinstance(m, nn.Module))
            m.to(device)

    def move_optimizer_to(optim, device):
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def _tmp_module_to(module, device):
        for p in module.parameters():
            p.tmp_data_orig = p.data
            p.data = p.data.to(device)

            # TODO: think about grad workflow. Probably it need to copy back only grads instead of params during training
            #if p.grad is not None:
            #    p.tmp_grad_data_orig = p.grad.data
            #    p.grad.data = p.grad.data.to(device)

        for b in module.buffers():
            b.tmp_data_orig = b.data
            b.data = b.data.to(device)

        return module

    def _tmp_module_reset(module):
        for p in module.parameters():
            assert(p.tmp_data_orig is not None)
            p.data = p.tmp_data_orig
            p.tmp_data_orig = None

            #if p.grad is not None:
            #    assert(p.tmp_grad_data_orig is not None)
            #    p.grad.data = p.tmp_grad_data_orig
            #    p.tmp_grad_data_orig = None

        for b in module.buffers():
            assert(b.tmp_data_orig is not None)
            b.data = b.tmp_data_orig
            b.tmp_data_orig = None

        return module

    def tmp_modules_to(mods, dev):
        for m in mods:
            modules._tmp_module_to(m, dev)

    def tmp_modules_reset(mods):
        for m in mods:
            modules._tmp_module_reset(m)

    def get_module_device(module):
        try:
            return next(module.parameters()).device
        except:
            return next(module.buffers()).device

    def print_cuda_mem():
        mem = torch.cuda.memory_allocated('cuda') / 1024**2
        print(f"... GPU={mem:.2f} MB")

#Static object
class runner:
    def gen_msg(brunner, stop_token, enc, check_cancel=None):
        response_tokenized = []
        for i in range(10000):
            # Check if cancellation was requested
            if check_cancel is not None and check_cancel():
                return []
            tokens, _ = brunner.step(enc, skip_backward=True, verbose=True)
            tokens = tokens.argmax(dim=-1)
            t = int(tokens[-1])
            enc['input_ids'].append(t)
            enc['attention_mask'].append(1)
            enc['labels'].append(t)  # For future continuous learning

            # Decode and print the new token
            if t == stop_token:
                break
            response_tokenized.append(t)

        return response_tokenized

    def get_stop_token(tu):
        stop_token = tu.tokenizer(tu.afixes[5], add_special_tokens=False)["input_ids"]
        assert(len(stop_token) == 1)
        stop_token = stop_token[0]
        return stop_token
