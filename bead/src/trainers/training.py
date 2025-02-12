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

import os
import random
import time
import sys
import numpy as np
from tqdm.rich import tqdm

from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader

from src.utils import helper, loss, diagnostics


class EarlyStopping:
    """
    Class to perform early stopping during model training.
    Attributes:
        patience (int): The number of epochs to wait before stopping the training process if the
            validation loss doesn't improve.
        min_delta (float): The minimum difference between the new loss and the previous best loss
            for the new loss to be considered an improvement.
        counter (int): Counts the number of times the validation loss hasn't improved.
        best_loss (float): The best validation loss observed so far.
        early_stop (bool): Flag that indicates whether early stopping criteria have been met.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience  # Nr of times we allow val. loss to not improve before early stopping
        self.min_delta = min_delta  # min(new loss - best loss) for new loss to be considered improvement
        self.counter = 0  # counts nr of times val_loss dosent improve
        self.best_loss = None
        self.early_stop = False

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss

        elif self.best_loss - train_loss > self.min_delta:
            self.best_loss = train_loss
            self.counter = 0  # Resets if val_loss improves

        elif self.best_loss - train_loss < self.min_delta:
            self.counter += 1

            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early Stopping")
                self.early_stop = True


class LRScheduler:
    """
    A learning rate scheduler that adjusts the learning rate of an optimizer based on the training loss.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        patience (int): The number of epochs with no improvement in training loss after which the learning rate
            will be reduced.
        min_lr (float, optional): The minimum learning rate that can be reached (default: 1e-6).
        factor (float, optional): The factor by which the learning rate will be reduced (default: 0.1).
    Attributes:
        lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The PyTorch learning rate scheduler that
            actually performs the adjustments.
    Example usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = LRScheduler(optimizer, patience=3, min_lr=1e-6, factor=0.5)
        for epoch in range(num_epochs):
            train_loss = train(model, train_data_loader)
            lr_scheduler(train_loss)
            # ...
    """

    def __init__(self, optimizer, patience, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        # Maybe add if statements for selectment of lr schedulers
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, loss):
        self.lr_scheduler.step(loss)


def fit(
    config,
    model,
    train_dl,
    reg_param,
    optimizer,
    latent_dim,
    n_dimensions,
):
    """This function trains the model on the train set. It computes the losses and does the backwards propagation, and updates the optimizer as well.
    Args:
        model (modelObject): The model you wish to train
        train_dl (torch.DataLoader): Defines the batched data which the model is trained on
        model_children (list): List of model parameters
        reg_param (float): Determines proportionality constant to balance different components of the loss.
        optimizer (torch.optim): Chooses optimizer for gradient descent.
        n_dimensions (int): Number of dimensions.
    Returns:
        list, model object: Training loss and trained model
    """
    # Extract model parameters
    model_children = list(model.children())

    model.train()

    running_loss = 0.0
    device = helper.get_device()

    for idx, inputs in enumerate(tqdm(train_dl, desc="Training: ")):
        inputs = inputs.to(device)

        # Set the gradients to zero
        optimizer.zero_grad()

        # Compute the predicted outputs from the input data
        out = helper.call_forward(model, inputs)

        if (
            hasattr(config, "custom_loss_function")
            and config.custom_loss_function == "loss_function_swae"
        ):
            z = model.encode(inputs)
            loss, mse_loss, l1_loss = loss.loss_function_swae(
                inputs, z, reconstructions, latent_dim
            )
        else:
            # Compute how far off the prediction is
            loss, mse_loss, l1_loss = loss.mse_sum_loss_l1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=reg_param,
                validate=True,
            )

        # Compute the loss-gradient with
        loss.backward()

        # Update the optimizer
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / (idx + 1)
    print(f"# Finished. Training Loss: {loss:.6f}")
    return epoch_loss, mse_loss, l1_loss, model


def validate(config, model, test_dl, reg_param):
    """Function used to validate the training. Not necessary for doing compression, but gives a good indication of wether the model selected is a good fit or not.
    Args:
        model (modelObject): Defines the model one wants to validate. The model used here is passed directly from `fit()`.
        test_dl (torch.DataLoader): Defines the batched data which the model is validated on
        model_children (list): List of model parameters
        reg_param (float): Determines proportionality constant to balance different components of the loss.
    Returns:
        float: Validation loss
    """
    # Extract model parameters
    model_children = list(model.children())

    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(test_dl, desc="Validating: ")):

            out = helper.call_forward(model, inputs)

            loss, _, _ = loss.mse_sum_loss_l1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=reg_param,
                validate=True,
            )
            running_loss += loss.item()

    epoch_loss = running_loss / (idx + 1)
    print(f"# Finished. Validation Loss: {loss:.6f}")
    return epoch_loss


def seed_worker(worker_id):
    """PyTorch implementation to fix the seeds
    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    model,
    events_train,
    jets_train,
    constituents_train,
    events_val,
    jets_val,
    constituents_val,
    output_path,
    config,
):
    """Does the entire training loop by calling the `fit()` and `validate()`. Appart from this, this is the main function where the data is converted
        to the correct type for it to be trained, via `torch.Tensor()`. Furthermore, the batching is also done here, based on `config.batch_size`,
        and it is the `torch.utils.data.DataLoader` doing the splitting.
        Applying either `EarlyStopping` or `LR Scheduler` is also done here, all based on their respective `config` arguments.
        For reproducibility, the seeds can also be fixed in this function.
    Args:
        model (modelObject): The model you wish to train
        data (Tuple): Tuple containing the training and validation data
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs
    Returns:
        modelObject: fully trained model ready to perform compression and decompression
    """

    if verbose:
        print("Events - Training set size:         ", events_train.size(0))
        print("Events - Validation set size:       ", events_val.size(0))
        print("Jets - Training set size:           ", jets_train.size(0))
        print("Jets - Validation set size:         ", jets_val.size(0))
        print("Constituents - Training set size:   ", constituents_train.size(0))
        print("Constituents - Validation set size: ", constituents_val.size(0))

    # Get the device and move tensors to the device
    device = helper.get_device()
    (
        events_train,
        jets_train,
        constituents_train,
        events_val,
        jets_val,
        constituents_val,
    ) = [
        x.to(device)
        for x in [
            events_train,
            jets_train,
            constituents_train,
            events_val,
            jets_val,
            constituents_val,
        ]
    ]
    model = model.to(device)
    if verbose:
        print(f"Device used for training: {device}")
        print(f"Inputs and model moved to device")

    # Pushing input data into the torch-DataLoader object and combines into one DataLoader object (a basic wrapper
    # around several DataLoader objects).
    if verbose:
        print(
            "Loading data into DataLoader and using batch size of ", config.batch_size
        )

    if config.deterministic_algorithm:
        if config.verbose:
            print("Deterministic algorithm is set to True")
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(0)

        train_dl = [
            DataLoader(
                ds,
                batch_size=config.batch_size,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
            )
            for ds in [events_train, jets_train, constituents_train]
        ]
        valid_dl = [
            DataLoader(
                ds,
                batch_size=config.batch_size,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
            )
            for ds in [events_val, jets_val, constituents_val]
        ]
    else:
        train_dl = [
            DataLoader(ds, batch_size=config.batch_size, shuffle=False, drop_last=True)
            for ds in [events_train, jets_train, constituents_train]
        ]
        valid_dl = [
            DataLoader(ds, batch_size=config.batch_size, shuffle=False, drop_last=True)
            for ds in [events_val, jets_val, constituents_val]
        ]
    # Unpacking the DataLoader lists
    train_dl_events, train_dl_jets, train_dl_constituents = train_dl
    val_dl_events, val_dl_jets, val_dl_constituents = valid_dl

    # Select Optimizer
    try:
        optimizer = helper.get_optimizer(
            config.optimizer, model.parameters(), lr=config.lr
        )
    except ValueError as e:
        print(e)

    # Activate early stopping
    if config.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience, min_delta=config.min_delta
        )  # Changes to patience & min_delta can be made in configs

    # Activate LR Scheduler
    if config.lr_scheduler:
        lr_scheduler = LRScheduler(
            optimizer=optimizer, patience=config.lr_scheduler_patience
        )

    # Training and Validation of the model
    train_loss = []
    val_loss = []
    start = time.time()

    # Registering hooks for activation extraction
    if config.activation_extraction:
        hooks = model.store_hooks()

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1} / {config.epochs}")

        train_epoch_loss, mse_loss_fit, regularizer_loss_fit, trained_model = fit(
            config=config,
            model=model,
            train_dl=train_dl,
            reg_param=config.reg_param,
            optimizer=optimizer,
            latent_dim=config.latent_space_size,
            n_dimensions=config.data_dimension,
        )
        train_loss.append(train_epoch_loss)

        if config.train_size:
            val_epoch_loss = validate(
                model=trained_model,
                test_dl=valid_dl,
                reg_param=config.reg_param,
            )
            val_loss.append(val_epoch_loss)
        else:
            val_epoch_loss = train_epoch_loss
            val_loss.append(val_epoch_loss)

        if config.lr_scheduler:
            lr_scheduler(val_epoch_loss)
        if config.early_stopping:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

        ## Implementation to save models & values after every N config.epochs, where N is stored in 'config.intermittent_saving_patience':
        if config.intermittent_model_saving:
            if epoch % config.intermittent_saving_patience == 0:
                path = os.path.join(output_path, "models", f"model_{epoch}.pt")
                helper.model_saver(model, path)

    end = time.time()

    # Saving activations values
    if config.activation_extraction:
        activations = diagnostics.dict_to_square_matrix(model.get_activations())
        model.detach_hooks(hooks)
        np.save(os.path.join(project_path, "activations.npy"), activations)

    if verbose:
        print(f"Training the model took {(end - start) / 60:.3} minutes")
    np.save(
        os.path.join(project_path, "loss_data.npy"), np.array([train_loss, val_loss])
    )

    if config.model_type == "convolutional":
        final_layer = model.get_final_layer_dims()
        np.save(os.path.join(project_path, "final_layer.npy"), np.array(final_layer))

    return trained_model
