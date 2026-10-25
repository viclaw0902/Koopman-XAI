from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


class Rescale(torch.autograd.Function):
    @staticmethod
    def forward(context, current, reference, kind):
        if kind == "gelu":
            result = functional.gelu(current)
            baseline = functional.gelu(reference)
            derivative = 0.5 * (1 + torch.erf(current / math.sqrt(2))) + current * torch.exp(-current.square() / 2) / math.sqrt(2 * math.pi)
        elif kind == "sigmoid":
            result = current.sigmoid()
            baseline = reference.sigmoid()
            derivative = result * (1 - result)
        elif kind == "tanh":
            result = current.tanh()
            baseline = reference.tanh()
            derivative = 1 - result.square()
        else:
            raise ValueError(kind)
        difference = current - reference
        denominator = torch.where(difference.abs() > 1e-7, difference, torch.ones_like(difference))
        multiplier = torch.where(difference.abs() > 1e-7, (result - baseline) / denominator, derivative)
        context.save_for_backward(multiplier)
        return result

    @staticmethod
    def backward(context, gradient):
        multiplier, = context.saved_tensors
        return gradient * multiplier, None, None


class Product(torch.autograd.Function):
    @staticmethod
    def forward(context, left, right, reference_left, reference_right):
        context.save_for_backward((right + reference_right) / 2, (left + reference_left) / 2)
        return left * right

    @staticmethod
    def backward(context, gradient):
        left_multiplier, right_multiplier = context.saved_tensors
        return gradient * left_multiplier, gradient * right_multiplier, None, None


def nonlinear(current, reference, kind):
    result = Rescale.apply(current, reference, kind)
    if kind == "gelu":
        baseline = functional.gelu(reference)
    elif kind == "sigmoid":
        baseline = reference.sigmoid()
    else:
        baseline = reference.tanh()
    return result, baseline


def linear_pair(layer, current, reference):
    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.BatchNorm1d)):
        return layer(current), layer(reference)
    if isinstance(layer, nn.GELU):
        return nonlinear(current, reference, "gelu")
    if isinstance(layer, nn.Dropout):
        if layer.training:
            raise ValueError("DeepLIFT requires evaluation mode")
        return current, reference
    raise TypeError(type(layer).__name__)


def paired_forward(model, windows, baseline):
    if model.training:
        raise ValueError("Classifier must be frozen in evaluation mode")
    current, reference = windows, baseline.detach()
    for layer in model.conv:
        current, reference = linear_pair(layer, current, reference)
    current, reference = current.transpose(1, 2), reference.transpose(1, 2)
    hidden = torch.zeros(len(windows), 64, device=windows.device, dtype=windows.dtype)
    reference_hidden = hidden.clone()
    recurrent = model.gru
    for step in range(current.shape[1]):
        input_gates = functional.linear(current[:, step], recurrent.weight_ih_l0, recurrent.bias_ih_l0).chunk(3, dim=1)
        hidden_gates = functional.linear(hidden, recurrent.weight_hh_l0, recurrent.bias_hh_l0).chunk(3, dim=1)
        reference_input = functional.linear(reference[:, step], recurrent.weight_ih_l0, recurrent.bias_ih_l0).chunk(3, dim=1)
        reference_gates = functional.linear(reference_hidden, recurrent.weight_hh_l0, recurrent.bias_hh_l0).chunk(3, dim=1)
        reset, reference_reset = nonlinear(input_gates[0] + hidden_gates[0], reference_input[0] + reference_gates[0], "sigmoid")
        update, reference_update = nonlinear(input_gates[1] + hidden_gates[1], reference_input[1] + reference_gates[1], "sigmoid")
        candidate_product = Product.apply(reset, hidden_gates[2], reference_reset, reference_gates[2])
        reference_product = reference_reset * reference_gates[2]
        candidate, reference_candidate = nonlinear(input_gates[2] + candidate_product, reference_input[2] + reference_product, "tanh")
        first = Product.apply(1 - update, candidate, 1 - reference_update, reference_candidate)
        second = Product.apply(update, hidden, reference_update, reference_hidden)
        hidden = first + second
        reference_hidden = (1 - reference_update) * reference_candidate + reference_update * reference_hidden
    for layer in model.head:
        hidden, reference_hidden = linear_pair(layer, hidden, reference_hidden)
    return hidden, reference_hidden


def explain(model, endpoint, baseline, target):
    window = endpoint.detach().unsqueeze(0).requires_grad_(True)
    logits, reference_logits = paired_forward(model, window, baseline.unsqueeze(0))
    multipliers = torch.autograd.grad(logits[0, target], window)[0]
    attribution = multipliers[0] * (endpoint - baseline)
    difference = logits[0, target] - reference_logits[0, target]
    error = (attribution.sum() - difference).abs()
    return attribution.detach(), {"completeness_absolute_error": float(error), "score_difference": float(difference.detach()),
                                  "implementation": "DeepLIFT Rescale for unary activations; symmetric RevealCancel/Shapley product rule for explicit GRU gate products"}
