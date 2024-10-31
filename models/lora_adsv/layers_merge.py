import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from copy import deepcopy

COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks", "layer"]

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


@dataclass
class LoraConfig:

    lora_moe_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    top_k: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_moe_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    merge_weights: bool = field(default=False, metadata={"help": "Whether to allow merge lora updates"})


class LoraModel(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter()

    def add_adapter(self, ):
        self._find_and_replace()

        if self.peft_config.inference_mode:
            _freeze_adapter(self.model, "lora")

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_new_module(self, lora_config, target, layer_index, linear_type):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "lora_moe_r": lora_config.lora_moe_r,
            "lora_moe_alpha": lora_config.lora_moe_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "top_k": lora_config.top_k[layer_index],
            "merge_weights": lora_config.merge_weights,
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False

        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )

        if linear_type == "Linear":
            new_module = Linear(in_features, out_features, bias=bias, **kwargs)
        elif linear_type == "MergedLinear":
            kwargs.update({"enable_lora": [True, True, True]})
            new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
        else:
            raise ValueError

        return new_module

    def _find_and_replace(self, ):
        lora_config = self.peft_config

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            layer_index = int(key.split(".")[1])

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if "qkv" in key:
                new_module = self._create_new_module(lora_config, target, layer_index, linear_type="MergedLinear")

            else:
                new_module = self._create_new_module(lora_config, target, layer_index, linear_type="Linear")

            self._replace_module(parent, target_name, new_module, target, layer_index)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module, layer_index):

        setattr(parent_module, child_name, new_module)

        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "router" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.merge()

    def merge_multi_adapter(self, routing_weights, selected_experts):
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.merge_multi_expert(routing_weights, selected_experts)

    def merge_specified_adapter(self, selected_experts):
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.merge_specified_expert(selected_experts)

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.unmerge()

    def set_svd_trim(self, svd_trim_names, svd_trim_layers, trim_p, reinit=True):
        if reinit:
            for module in self.model.modules():
                if isinstance(module, LoRALayer):
                    module.trim_flag = False

        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                trim_flag = True
                if type(svd_trim_layers) == float or type(svd_trim_layers) == int:
                    if svd_trim_layers != -1 and not name.startswith(f"model.blocks.{svd_trim_layers}."):
                        trim_flag = False
                elif type(svd_trim_layers) == list:
                    flag = any([name.startswith(f"model.blocks.{str(l)}.") for l in svd_trim_layers])
                    if not flag:
                        trim_flag = False
                else:
                    raise ValueError(f"{svd_trim_layers}")

                if trim_flag:
                    if type(svd_trim_names) == list:
                        trim_flag = any([name.endswith(f"{converted_name}") for converted_name in svd_trim_names])
                    elif type(svd_trim_names) == str:
                        trim_flag = name.endswith(f"{svd_trim_names}")
                    else:
                        raise ValueError(f"{svd_trim_names}")

                if trim_flag:
                    module.trim_flag = True
                    module.trim_p = trim_p

    def svd_trim(self,):
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.svd_trim()

def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and 'router' not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def mark_only_lora_B_as_trainable(model: nn.Module, bias: str = "none") -> None:
    print('*******************', 'ONLY TRAIN EXPERTS WITHOUT ROUTER', '*******************')
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

class LoRALayer():
    def __init__(
        self,
        merge_weights: bool,
    ):

        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        lora_moe_r = [0],
        lora_moe_alpha = [1],
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        top_k: int = 2,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.scaling = {}
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.router = nn.ModuleDict({})
        self.number_experts = len(lora_moe_r)
        for idx, r in enumerate(lora_moe_r):
            if r > 0:
                adapter_name_moe = str(idx)
                self.lora_A.update(nn.ParameterDict({adapter_name_moe: nn.Parameter(self.weight.new_zeros((r, in_features)))}))
                self.lora_B.update(nn.ParameterDict({adapter_name_moe: nn.Parameter(self.weight.new_zeros((out_features, r)))}))
                self.scaling[adapter_name_moe] = lora_moe_alpha[idx] / r
        self.weight.requires_grad = False
        self.reset_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        self.ori_weight = None
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.trim_flag = False
        self.trim_p = 0

    def reset_lora_parameters(self):
        nn.Linear.reset_parameters(self)

        for idx in range(self.number_experts):
            adapter_name_moe = str(idx)
            if adapter_name_moe in self.lora_A.keys():

                nn.init.kaiming_uniform_(self.lora_A[adapter_name_moe], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name_moe])

    def merge_multi_expert(self, routing_weights, selected_experts):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            raise RuntimeError("Already merged. Unmerge First")

        if self.merge_weights and not self.merged:
            self.ori_weight = deepcopy(self.weight)

            for alpha, idx in zip(routing_weights, selected_experts):
                adapter_name_moe = str(idx.item())
                expert_weight = T(self.lora_B[adapter_name_moe] @ self.lora_A[adapter_name_moe]) * self.scaling[adapter_name_moe]
                self.weight.data += alpha * expert_weight
                self.merged = True

    @torch.no_grad()
    def merge_specified_expert(self, selected_experts):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            raise RuntimeError("Already merged. Unmerge First")

        if self.merge_weights and not self.merged:
            self.ori_weight = deepcopy(self.weight)

            if not isinstance(selected_experts, list):
                selected_experts = [selected_experts]
            deno = len(selected_experts)
            for idx in selected_experts:
                adapter_name_moe = str(idx)
                expert_weight = T(self.lora_B[adapter_name_moe] @ self.lora_A[adapter_name_moe]) * self.scaling[adapter_name_moe]
                self.weight.data += expert_weight / deno
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merge_weights and self.merged:
            self.weight.data.copy_(self.ori_weight.data)
            self.merged = False

    @torch.no_grad()
    def svd_trim(self):
        if self.trim_flag:
            desired_rank = int(min(self.weight.shape[0], self.weight.shape[1]) * self.trim_p)
            weight_copy = deepcopy(self.weight)
            results = torch.svd_lowrank(weight_copy.type(torch.float32), q=desired_rank, niter=2)
            self.weight = torch.nn.Parameter(results[0] @ torch.diag(results[1]) @ results[2].T)

    def forward(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        previous_dtype = x.dtype

        assert self.merged == True, "should merge weight first"
        results = F.linear(x, T(self.weight), bias=self.bias)
        results = results.to(previous_dtype)

        return results


class MergedLinear(nn.Linear, LoRALayer):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        lora_moe_r = [0],
        lora_moe_alpha = [1],
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [True, True, True],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        top_k: int = 2,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.scaling = {}
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.router = nn.ModuleDict({})
        self.number_experts = len(lora_moe_r)

        for idx, r in enumerate(lora_moe_r):
            if r > 0:
                adapter_name_moe = str(idx)
                self.lora_A.update(nn.ParameterDict({adapter_name_moe: nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))}))
                self.lora_B.update(nn.ParameterDict({adapter_name_moe: nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))}))
                self.scaling[adapter_name_moe] = lora_moe_alpha[idx] / r

        self.weight.requires_grad = False

        self.lora_ind = self.weight.new_zeros(
            (out_features, ), dtype=torch.bool
        ).view(len(enable_lora), -1)
        self.lora_ind[enable_lora, :] = True
        self.lora_ind = self.lora_ind.view(-1)
        self.reset_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.ori_weight = None
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.trim_flag = False
        self.trim_p = 0

    def reset_lora_parameters(self):
        nn.Linear.reset_parameters(self)

        for idx in range(self.number_experts):
            adapter_name_moe = str(idx)
            if adapter_name_moe in self.lora_A.keys():

                nn.init.kaiming_uniform_(self.lora_A[adapter_name_moe], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name_moe])

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self, lora_A, lora_B):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            lora_A.unsqueeze(0),
            lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def merge_multi_expert(self, routing_weights, selected_experts):
        if self.merged:
            raise RuntimeError("Already merged. Unmerge First")

        if self.merge_weights and not self.merged:
            self.ori_weight = deepcopy(self.weight)

            for alpha, idx in zip(routing_weights, selected_experts):
                adapter_name_moe = str(idx.item())
                expert_weight = self.merge_AB(self.lora_A[adapter_name_moe], self.lora_B[adapter_name_moe]) * self.scaling[adapter_name_moe]
                self.weight.data += alpha * expert_weight
                self.merged = True

    @torch.no_grad()
    def merge_specified_expert(self, selected_experts):
        if self.merged:
            raise RuntimeError("Already merged. Unmerge First")

        if self.merge_weights and not self.merged:
            self.ori_weight = deepcopy(self.weight)

            if not isinstance(selected_experts, list):
                selected_experts = [selected_experts]
            deno = len(selected_experts)
            for idx in selected_experts:
                adapter_name_moe = str(idx)
                expert_weight = self.merge_AB(self.lora_A[adapter_name_moe], self.lora_B[adapter_name_moe]) * self.scaling[adapter_name_moe]
                self.weight.data += expert_weight / deno
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merge_weights and self.merged:
            self.weight.data.copy_(self.ori_weight.data)
            self.merged = False

    @torch.no_grad()
    def svd_trim(self):
        if self.trim_flag:
            desired_rank = int(min(self.weight.shape[0], self.weight.shape[1]) * self.trim_p)
            weight_copy = deepcopy(self.weight)
            results = torch.svd_lowrank(weight_copy.type(torch.float32), q=desired_rank, niter=2)
            self.weight = torch.nn.Parameter(results[0] @ torch.diag(results[1]) @ results[2].T)

    def forward(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        previous_dtype = x.dtype

        assert self.merged == True, "should merge weight first"
        results = F.linear(x, T(self.weight), bias=self.bias)
        results = results.to(previous_dtype)

        return results