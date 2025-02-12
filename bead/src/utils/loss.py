# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch.nn import functional
from tqdm import tqdm
from torch.nn import functional as F
from torch import distributions as dist


def loss_function_swae(
    inputs,
    z,
    reconstructions,
    latent_dim,
    reg_weight=100,
    wasserstein_deg=2.0,
    num_projections=2000,
    projection_dist="normal",
):
    batch_size = inputs.shape[0]
    bias_corr = batch_size * (batch_size - 1)
    reg_weight = reg_weight / bias_corr

    mse_sum = nn.MSELoss(reduction="sum")
    mse_loss = mse_sum(reconstructions, inputs)
    number_of_columns = inputs.shape[1]
    mse_sum_loss = mse_loss / number_of_columns
    # recons_loss_l1 = F.l1_loss(reconstructions, inputs)
    recons_loss_l1 = 0
    swd_loss = compute_swd(
        z, wasserstein_deg, reg_weight, latent_dim, num_projections, projection_dist
    )
    loss = mse_sum_loss + recons_loss_l1 + swd_loss
    SWD = swd_loss
    return loss, mse_sum_loss, SWD


def compute_swd(z, p, reg_weight, latent_dim, num_projections, proj_dist):
    prior_z = torch.randn_like(z)  # [N x D]
    device = z.device

    proj_matrix = (
        get_random_projections(proj_dist, latent_dim, num_samples=num_projections)
        .transpose(0, 1)
        .to(device)
    )

    latent_projections = z.matmul(proj_matrix)  # [N x S]
    prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise distance
    w_dist = (
        torch.sort(latent_projections.t(), dim=1)[0]
        - torch.sort(prior_projections.t(), dim=1)[0]
    )
    w_dist = w_dist.pow(p)
    return reg_weight * w_dist.mean()


def get_random_projections(proj_dist, latent_dim, num_samples):
    if proj_dist == "normal":
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == "cauchy":
        rand_samples = (
            dist.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
            .sample((num_samples, latent_dim))
            .squeeze()
        )
    else:
        raise ValueError("Unknown projection distribution.")

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
    return rand_proj  # [S x D]


def mse_loss_emd_l1(model_children, true_data, reconstructed_data, reg_param, validate):
    """
    Computes a sparse loss function consisting of three terms: the Earth Mover's Distance (EMD) loss between the
    true and reconstructed data, the mean squared error (MSE) loss between the reconstructed and true data, and a
    L1 regularization term on the output of a list of model children.
    
    Args: model_children (list): List of PyTorch modules representing the model architecture to be regularized.
    true_data (torch.Tensor): The ground truth data, with shape (batch_size, num_features). reconstructed_data (
    torch.Tensor): The reconstructed data, with shape (batch_size, num_features). reg_param (float): The weight of
    the L1 regularization term in the loss function. validate (bool): If True, returns only the EMD loss. If False,
    computes the full loss with the L1 regularization term.
    
    Returns:
        If validate is False, returns a tuple with three elements:
        - loss (torch.Tensor): The full sparse loss function, with shape ().
        - emd_loss (float): The EMD loss between the true and reconstructed data.
        - l1_loss (torch.Tensor): The L1 regularization term on the output of the model children.

        If validate is True, returns only the EMD loss as a float.
    """
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)
    wasserstein_distance_list = [
        wasserstein_distance(
            true_data.detach().numpy()[i, :], reconstructed_data.detach().numpy()[i, :]
        )
        for i in range(len(true_data))
    ]
    emd_loss = sum(wasserstein_distance_list)

    l1_loss = torch.Tensor(0)
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = model_children[i](values)
            l1_loss += torch.mean(torch.abs(values))

        loss = emd_loss + mse_loss + reg_param * l1_loss
        return loss, emd_loss, l1_loss
    else:
        return emd_loss


def mse_loss_l1(model_children, true_data, reconstructed_data, reg_param, validate):
    # This function is a modified version of the original function by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    """
    Computes a sparse loss function consisting of two terms: the mean squared error (MSE) loss between the
    reconstructed and true data, and a L1 regularization term on the output of a list of model children.
    
    Args: model_children (list): List of PyTorch modules representing the model architecture to be regularized.
    true_data (torch.Tensor): The ground truth data, with shape (batch_size, num_features). reconstructed_data (
    torch.Tensor): The reconstructed data, with shape (batch_size, num_features). reg_param (float): The weight of
    the L1 regularization term in the loss function. validate (bool): If True, returns only the MSE loss. If False,
    computes the full loss with the L1 regularization term.
    
    Returns:
        If validate is False, returns a tuple with three elements:
        - loss (torch.Tensor): The full sparse loss function, with shape ().
        - mse_loss (float): The MSE loss between the true and reconstructed data.
        - l1_loss (torch.Tensor): The L1 regularization term on the output of the model children.

        If validate is True, returns a tuple with three elements:
        - mse_loss (torch.Tensor): The MSE loss between the true and reconstructed data.
        - 0.
        - 0.
    """
    mse = nn.MSELoss()
    mse_loss = mse(reconstructed_data, true_data)

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = functional.relu(model_children[i](values))
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_loss + reg_param * l1_loss
        return loss, mse_loss, l1_loss
    else:
        return mse_loss, 0, 0


def mse_sum_loss_l1(model_children, true_data, reconstructed_data, reg_param, validate):
    """
    Computes the sum of mean squared error (MSE) loss and L1 regularization loss.
    Args:
        model_children (list): List of PyTorch modules representing the encoder network.
        true_data (tensor): Ground truth tensor of shape (batch_size, input_size).
        reconstructed_data (tensor): Reconstructed tensor of shape (batch_size, input_size).
        reg_param (float): Regularization parameter for L1 loss.
        validate (bool): Whether to return only MSE loss or both MSE and L1 losses.
    Returns:
        If validate is False:
            loss (tensor): Total loss consisting of MSE loss and L1 regularization loss.
            mse_sum_loss (tensor): Mean squared error loss.
            l1_loss (tensor): L1 regularization loss.
        If validate is True:
            mse_sum_loss (tensor): Mean squared error loss.
            0 (int): Placeholder for MSE loss since it is not calculated during validation.
            0 (int): Placeholder for L1 loss since it is not calculated during validation.
    """
    mse_sum = nn.MSELoss(reduction="sum")
    mse_loss = mse_sum(reconstructed_data, true_data)
    number_of_columns = true_data.shape[1]

    mse_sum_loss = mse_loss / number_of_columns

    l1_loss = 0
    values = true_data
    if not validate:
        for i in range(len(model_children)):
            values = functional.relu(model_children[i](values))
            l1_loss += torch.mean(torch.abs(values))

        loss = mse_sum_loss + reg_param * l1_loss
        return loss, mse_sum_loss, l1_loss
    else:
        return mse_sum_loss, 0, 0


# Accuracy function still WIP. Not working properly.
# Probably has to do with total_correct counter.


def accuracy(model, dataloader):
    """
    Computes the accuracy of a PyTorch model on a given dataset.
    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader object containing the dataset to evaluate on.
    Returns:
        accuracy_frac (float): The fraction of correctly classified instances in the dataset.
    """
    print("Accuracy")
    model.eval()

    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            x, _ = data
            classifications = torch.argmax(x)

            correct_pred = torch.sum(classifications == x).item()

            total_correct += correct_pred
            total_instances += len(x)

    accuracy_frac = round(total_correct / total_instances, 3)
    print(accuracy_frac)
    return accuracy_frac
