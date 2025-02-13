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

import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def fit(
    config,
    model,
    train_dl,
    loss_fn,
    reg_param,
    optimizer,
):
    """This function trains the model on the train set. It computes the losses and does the backwards propagation, and updates the optimizer as well.
    Args:
        config (dataClass): Base class selecting user inputs
        model (modelObject): The model you wish to train
        train_dl (torch.DataLoader): Defines the batched data which the model is trained on
        loss (lossObject): Defines the loss function used to train the model
        reg_param (float): Determines proportionality constant to balance different components of the loss.
        optimizer (torch.optim): Chooses optimizer for gradient descent.
    Returns:
        list, model object: Training losses, Epoch_loss and trained model
    """
    # Extract model parameters
    parameters = model.parameters()

    model.train()

    running_loss = 0.0

    for idx, inputs in enumerate(tqdm(train_dl)):
        # Set the gradients to zero
        optimizer.zero_grad()

        # Compute the predicted outputs from the input data
        out = helper.call_forward(model, inputs)
        recon, mu, logvar, ldj, z0, zk = out

        # Compute the loss
        losses = loss_fn.calculate(
            recon=recon, target=inputs, mu=mu, logvar=logvar, parameters=parameters, log_det_jacobian=0
        )
        
        loss, *_ = losses
        
        # Compute the loss-gradient with
        loss.backward()

        # Update the optimizer
        optimizer.step()

        running_loss += loss

    epoch_loss = running_loss / (idx + 1)
    print(f"# Training Loss: {epoch_loss:.6f}")
    return losses, epoch_loss, model


def validate(config, model, test_dl, loss_fn, reg_param):
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
    parameters = model.parameters()

    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(test_dl)):

            out = helper.call_forward(model, inputs)
            recon, mu, logvar, ldj, z0, zk = out

            # Compute the loss
            losses = loss_fn.calculate(
                recon=recon, target=inputs, mu=mu, logvar=logvar, parameters=parameters, log_det_jacobian=0
            )

            loss, *_ = losses

            running_loss += loss

        epoch_loss = running_loss / (idx + 1)
        print(f"# Validation Loss: {epoch_loss:.6f}")
    return losses, epoch_loss


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
    verbose: bool = False,
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
    # Reshape tensors to pass to conv layers
    events_train, jets_train, constituents_train, events_val, jets_val, constituents_val = [
        x.unsqueeze(1).float() for x in [events_train, jets_train, constituents_train, events_val, jets_val, constituents_val]
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

        train_dl_list = [
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
        valid_dl_list = [
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
        train_dl_list = [
            DataLoader(ds, batch_size=config.batch_size, shuffle=False, drop_last=True)
            for ds in [events_train, jets_train, constituents_train]
        ]
        valid_dl_list = [
            DataLoader(ds, batch_size=config.batch_size, shuffle=False, drop_last=True)
            for ds in [events_val, jets_val, constituents_val]
        ]
    # Unpacking the DataLoader lists
    train_dl_events, train_dl_jets, train_dl_constituents = train_dl_list
    val_dl_events, val_dl_jets, val_dl_constituents = valid_dl_list

    if config.model_name == "pj_ensemble":
        if verbose:
            print("Model is an ensemble model")
    else:
        if config.input_level == "event":
            train_dl = train_dl_events
            valid_dl = val_dl_events
        elif config.input_level == "jet":
            train_dl = train_dl_jets
            valid_dl = val_dl_jets
        elif config.input_level == "constituent":
            train_dl = train_dl_constituents
            valid_dl = val_dl_constituents
        if verbose:
            print(f"Input data is of {config.input_level} level")

    # Select Loss Function
    try:
        loss_object = helper.get_loss(config.loss_function)
        loss_fn = loss_object(config=config)
        if verbose:
            print(f"Loss Function: {config.loss_function}")
    except ValueError as e:
        print(e)

    # Select Optimizer
    try:
        optimizer = helper.get_optimizer(
            config.optimizer, model.parameters(), lr=config.lr
        )
        if verbose:
            print(f"Optimizer: {config.optimizer}")
    except ValueError as e:
        print(e)

    # Activate early stopping
    if config.early_stopping:
        if verbose:
            print("Early stopping is activated with patience of ", config.early_stopping_patience)
        early_stopper = helper.EarlyStopping(
            patience=config.early_stopping_patience, min_delta=config.min_delta
        )  # Changes to patience & min_delta can be made in configs

    # Activate LR Scheduler
    if config.lr_scheduler:
        if verbose:
            print("Learning rate scheduler is activated with patience of ", config.lr_scheduler_patience)
        lr_scheduler = helper.LRScheduler(
            optimizer=optimizer, patience=config.lr_scheduler_patience
        )

    # Training and Validation of the model
    train_loss_data = []
    val_loss_data = []
    train_loss = []
    val_loss = []
    start = time.time()

    # Registering hooks for activation extraction
    if config.activation_extraction:
        hooks = model.store_hooks()

    if verbose:
        print(f"Beginning training for {config.epochs} epochs")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1} / {config.epochs}")

        train_losses, train_epoch_loss, model = fit(
            config=config,
            model=model,
            train_dl=train_dl,
            loss_fn=loss_fn,
            reg_param=config.reg_param,
            optimizer=optimizer,
        )
        train_loss.append(train_epoch_loss.item())
        train_loss_data.append(train_losses)

        if 1-config.train_size:
            val_losses, val_epoch_loss = validate(
                config=config,
                model=model,
                test_dl=valid_dl,
                loss_fn=loss_fn,
                reg_param=config.reg_param,
            )
            val_loss.append(val_epoch_loss.item())
            val_loss_data.append(val_losses)
        else:
            val_epoch_loss = train_epoch_loss
            val_losses = train_losses
            val_loss.append(val_epoch_loss)
            val_loss_data.append(val_losses)

        # Implementing LR Scheduler
        if config.lr_scheduler:
            lr_scheduler(val_epoch_loss)

        ## Implementation to save models & values after every N config.epochs, where N is stored in 'config.intermittent_saving_patience':
        if config.intermittent_model_saving:
            if epoch % config.intermittent_saving_patience == 0:
                path = os.path.join(output_path, "models", f"model_{epoch}.pt")
                helper.model_saver(model, path)

        # Implementing Early Stopping
        if config.early_stopping:
            early_stopper(val_epoch_loss)
            if early_stopper.early_stop:
                if verbose:
                    print("Early stopping activated at epoch ", epoch)
                break

    end = time.time()

    # Saving activations values
    if config.activation_extraction:
        activations = diagnostics.dict_to_square_matrix(model.get_activations())
        model.detach_hooks(hooks)
        np.save(os.path.join(project_path, "activations.npy"), activations)

    if verbose:
        print(f"Training the model took {(end - start) / 60:.3} minutes")
    
    # Save loss data
    # def extract_items(data):
    #     if isinstance(data, tuple):
    #         return tuple(extract_items(item) for item in data)
    #     elif isinstance(data, torch.Tensor):
    #         return data.item()
    #     else:
    #         raise TypeError("Unsupported type in tuple")

    # # Convert to Python scalars
    # converted_train_losses = [extract_items(tup) for tup in train_loss_data]
    # converted_val_losses = [extract_items(tup) for tup in val_loss_data]
    
    np.save(
        os.path.join(output_path, "results", "epoch_loss_data.npy"), np.array([train_loss, val_loss])
    )
    # np.save(os.path.join(output_path, "results", "train_loss_data.npy"), np.array(converted_train_losses))
    # np.save(os.path.join(output_path, "results", "val_loss_data.npy"), np.array(converted_val_losses))
    if verbose:
        print("Epoch loss data saved as [train_loss, val_loss] to path: ", os.path.join(output_path, "results", "epoch_loss_data.npy"))
        # print("Loss data saved as [train_losses, val_losses] to path: ", os.path.join(output_path, "results", "loss_data.npy"))

    return model
