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


class BaseLoss:
    """
    Base class for all loss functions.
    Each subclass must implement the calculate() method.
    """
    def __init__(self, config):
        self.config = config

    def calculate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the calculate() method.")

# ---------------------------
# Standard AE/VAE Losses
# ---------------------------
class ReconstructionLoss(BaseLoss):
    """
    Reconstruction loss for AE/VAE models.
    Supports both MSE and L1 losses based on configuration.
    
    Config parameters:
      - loss_type: 'mse' (default) or 'l1'
      - reduction: reduction method (default 'mean')
    """
    def __init__(self, config):
        super(ReconstructionLoss, self).__init__(config)
        self.loss_type = self.config.get("loss_type", "mse")
        self.reduction = self.config.get("reduction", "mean")

    def calculate(self, recon, target):
        if self.loss_type == "mse":
            loss = F.mse_loss(recon, target, reduction=self.reduction)
        elif self.loss_type == "l1":
            loss = F.l1_loss(recon, target, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {self.loss_type}")
        return loss

class KLDivergenceLoss(BaseLoss):
    """
    KL Divergence loss for VAE latent space regularization.
    
    Uses the formula:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    def __init__(self, config):
        super(KLDivergenceLoss, self).__init__(config)

    def calculate(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        batch_size = mu.size(0)
        return kl_loss / batch_size

class VAELoss(BaseLoss):
    """
    Total loss for VAE training.
    Combines reconstruction loss and KL divergence loss.
    
    Config parameters:
      - reconstruction: dict for ReconstructionLoss config.
      - kl: dict for KLDivergenceLoss config.
      - kl_weight: scaling factor for KL loss (default: 1.0)
    """
    def __init__(self, config):
        super(VAELoss, self).__init__(config)
        self.recon_loss_fn = ReconstructionLoss(config.get("reconstruction", {}))
        self.kl_loss_fn = KLDivergenceLoss(config.get("kl", {}))
        self.kl_weight = config.get("kl_weight", 1.0)

    def calculate(self, recon, target, mu, logvar):
        recon_loss = self.recon_loss_fn.calculate(recon, target)
        kl_loss = self.kl_loss_fn.calculate(mu, logvar)
        return recon_loss + self.kl_weight * kl_loss

# ---------------------------
# Advanced VAE+Flow Loss
# ---------------------------
class VAEFlowLoss(BaseLoss):
    """
    Loss for VAE models augmented with a normalizing flow.
    Includes the log_det_jacobian term from the flow transformation.
    
    Config parameters:
      - reconstruction: dict for ReconstructionLoss config.
      - kl: dict for KLDivergenceLoss config.
      - kl_weight: weight for the KL divergence term.
      - flow_weight: weight for the log_det_jacobian term.
    """
    def __init__(self, config):
        super(VAEFlowLoss, self).__init__(config)
        self.recon_loss_fn = ReconstructionLoss(config.get("reconstruction", {}))
        self.kl_loss_fn = KLDivergenceLoss(config.get("kl", {}))
        self.kl_weight = config.get("kl_weight", 1.0)
        self.flow_weight = config.get("flow_weight", 1.0)

    def calculate(self, recon, target, mu, logvar, log_det_jacobian):
        recon_loss = self.recon_loss_fn.calculate(recon, target)
        kl_loss = self.kl_loss_fn.calculate(mu, logvar)
        # Subtract the log-det term (maximizing likelihood).
        total_loss = recon_loss + self.kl_weight * kl_loss - self.flow_weight * log_det_jacobian
        return total_loss

# ---------------------------
# Contrastive Loss
# ---------------------------
class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss to cluster latent vectors by event generator.
    
    Config parameters:
      - margin: minimum distance desired between dissimilar pairs (default: 1.0)
    """
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__(config)
        self.margin = self.config.get("margin", 1.0)

    def calculate(self, latent, generator_flags):
        batch_size = latent.size(0)
        distances = torch.cdist(latent, latent, p=2)
        generator_flags = generator_flags.view(-1, 1)
        same_generator = (generator_flags == generator_flags.t()).float()
        pos_loss = same_generator * distances.pow(2)
        neg_loss = (1 - same_generator) * F.relu(self.margin - distances).pow(2)
        num_pairs = batch_size * (batch_size - 1)
        return (pos_loss.sum() + neg_loss.sum()) / num_pairs

# ---------------------------
# Earth Mover's Distance / Wasserstein Loss
# ---------------------------
class WassersteinLoss(BaseLoss):
    """
    Computes an approximation of the Earth Mover's Distance (Wasserstein Loss)
    between two 1D probability distributions.
    
    Assumes inputs are tensors of shape (batch_size, n) representing histograms or distributions.
    
    Config parameters:
      - dim: dimension along which to compute the cumulative sum (default: 1)
    """
    def __init__(self, config):
        super(WassersteinLoss, self).__init__(config)
        self.dim = self.config.get("dim", 1)

    def calculate(self, p, q):
        # Normalize if not already probability distributions
        p = p / (p.sum(dim=self.dim, keepdim=True) + 1e-8)
        q = q / (q.sum(dim=self.dim, keepdim=True) + 1e-8)
        p_cdf = torch.cumsum(p, dim=self.dim)
        q_cdf = torch.cumsum(q, dim=self.dim)
        loss = torch.mean(torch.abs(p_cdf - q_cdf))
        return loss

# ---------------------------
# Regularization Losses
# ---------------------------
class L1Regularization(BaseLoss):
    """
    Computes L1 regularization over model parameters.
    
    Config parameters:
      - weight: scaling factor for the L1 regularization (default: 1e-4)
    """
    def __init__(self, config):
        super(L1Regularization, self).__init__(config)
        self.weight = self.config.get("weight", 1e-4)

    def calculate(self, parameters):
        l1_loss = 0.0
        for param in parameters:
            l1_loss += torch.sum(torch.abs(param))
        return self.weight * l1_loss

class L2Regularization(BaseLoss):
    """
    Computes L2 regularization over model parameters.
    
    Config parameters:
      - weight: scaling factor for the L2 regularization (default: 1e-4)
    """
    def __init__(self, config):
        super(L2Regularization, self).__init__(config)
        self.weight = self.config.get("weight", 1e-4)

    def calculate(self, parameters):
        l2_loss = 0.0
        for param in parameters:
            l2_loss += torch.sum(param ** 2)
        return self.weight * l2_loss

# ---------------------------
# Energy Based Loss
# ---------------------------
class EnergyBasedLoss(BaseLoss):
    """
    An energy-based loss which encourages lower energy for normal samples and 
    higher energy (above a margin) for anomalous samples.
    
    Config parameters:
      - margin: margin threshold for anomaly energy (default: 1.0)
    """
    def __init__(self, config):
        super(EnergyBasedLoss, self).__init__(config)
        self.margin = self.config.get("margin", 1.0)

    def calculate(self, energy, labels):
        """
        Args:
            energy (Tensor): Predicted energy scores of shape (batch_size,).
            labels (Tensor): Binary labels of shape (batch_size,), with 0 indicating normal 
                             and 1 indicating anomaly.
                             
        Loss formulation:
            For normal samples (label=0): loss = energy^2 (encouraging low energy).
            For anomalous samples (label=1): loss = ReLU(margin - energy)^2 (encouraging high energy).
        """
        # Ensure labels are float tensors
        labels = labels.float()
        loss_normal = (1 - labels) * energy.pow(2)
        loss_anomaly = labels * F.relu(self.margin - energy).pow(2)
        return torch.mean(loss_normal + loss_anomaly)

# # ---------------------------
# # Example usage / Testing
# # ---------------------------
# if __name__ == "__main__":
#     # Example configuration dictionary (could be loaded from a file)
#     config = {
#         "reconstruction": {"loss_type": "mse", "reduction": "mean"},
#         "kl": {},
#         "kl_weight": 0.1,
#         "flow_weight": 0.5,
#         "margin": 1.0,
#         "dim": 1,
#         "weight": 1e-4
#     }
    
#     # Dummy data for demonstration
#     recon = torch.randn(4, 10)
#     target = torch.randn(4, 10)
#     mu = torch.randn(4, 5)
#     logvar = torch.randn(4, 5)
#     log_det_jacobian = torch.tensor(0.2)
#     latent = torch.randn(4, 5)
#     generator_flags = torch.tensor([0, 1, 0, 2])
    
#     # For WassersteinLoss assume two distributions (e.g., histograms)
#     p = torch.abs(torch.randn(4, 20))
#     q = torch.abs(torch.randn(4, 20))
    
#     # Example energy and labels for EnergyBasedLoss
#     energy = torch.randn(4).abs()
#     labels = torch.tensor([0, 1, 0, 1])
    
#     # Initialize losses
#     rec_loss_fn = ReconstructionLoss(config["reconstruction"])
#     kl_loss_fn = KLDivergenceLoss(config["kl"])
#     vae_loss_fn = VAELoss(config)
#     flow_loss_fn = VAEFlowLoss(config)
#     contrastive_loss_fn = ContrastiveLoss({"margin": config["margin"]})
#     wasserstein_loss_fn = WassersteinLoss({"dim": config["dim"]})
#     l1_reg_fn = L1Regularization({"weight": config["weight"]})
#     l2_reg_fn = L2Regularization({"weight": config["weight"]})
#     energy_loss_fn = EnergyBasedLoss({"margin": config["margin"]})
    
#     # Calculate and print losses
#     print("Reconstruction Loss:", rec_loss_fn.calculate(recon, target).item())
#     print("KL Divergence Loss:", kl_loss_fn.calculate(mu, logvar).item())
#     print("VAE Loss:", vae_loss_fn.calculate(recon, target, mu, logvar).item())
#     print("VAE+Flow Loss:", flow_loss_fn.calculate(recon, target, mu, logvar, log_det_jacobian).item())
#     print("Contrastive Loss:", contrastive_loss_fn.calculate(latent, generator_flags).item())
#     print("Wasserstein Loss:", wasserstein_loss_fn.calculate(p, q).item())
#     # For regularizations, assume we have a list of parameters (using recon tensor as a placeholder)
#     print("L1 Regularization Loss:", l1_reg_fn.calculate([recon]).item())
#     print("L2 Regularization Loss:", l2_reg_fn.calculate([recon]).item())
#     print("Energy Based Loss:", energy_loss_fn.calculate(energy, labels).item())

