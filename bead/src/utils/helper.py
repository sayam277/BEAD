# This file contains functions that help manipulate different artifacts as required
# in the pipeline. The functions in this file are used to manipulate data, models, and tensors.
import numpy as np
import os
import torch
import torch.nn as nn
from numpy import ndarray
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler, 
                                   PowerTransformer, QuantileTransformer, MaxAbsScaler)
from sklearn.base import BaseEstimator, TransformerMixin

from ..src.models import models


def get_device():
    """Returns the appropriate processing device. IF cuda is available it returns "cuda:0"
        Otherwise it returns "cpu"

    Returns:
        _type_: Device string, either "cpu" or "cuda:0"
    """
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    return device


def detach_device(tensor):
    """Detaches a given tensor to ndarray

    Args:
        tensor (torch.Tensor): The PyTorch tensor one wants to convert to a ndarray

    Returns:
        ndarray: Converted torch.Tensor to ndarray
    """
    return tensor.cpu().detach().numpy()


def convert_to_tensor(data):
    """Converts ndarray to torch.Tensors.

    Args:
        data (ndarray): The data you wish to convert from ndarray to torch.Tensor.

    Returns:
        torch.Tensor: Your data as a tensor
    """
    return torch.tensor(data, dtype=torch.float32)


def numpy_to_tensor(data):
    """Converts ndarray to torch.Tensors.

    Args:
        data (ndarray): The data you wish to convert from ndarray to torch.Tensor.

    Returns:
        torch.Tensor: Your data as a tensor
    """
    return torch.from_numpy(data)


def save_model(model, model_path: str) -> None:
    """Saves the models state dictionary as a `.pt` file to the given path.

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        None: Saved model state dictionary as `.pt` file.
    """
    torch.save(model.state_dict(), model_path)


def encoder_saver(model, model_path: str) -> None:
    """Saves the Encoder state dictionary as a `.pt` file to the given path

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        None: Saved encoder state dictionary as `.pt` file.
    """
    torch.save(model.encoder.state_dict(), model_path)


def decoder_saver(model, model_path: str) -> None:
    """Saves the Decoder state dictionary as a `.pt` file to the given path

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        None: Saved decoder state dictionary as `.pt` file.
    """
    torch.save(model.decoder.state_dict(), model_path)


def model_init(model_name: str, init: str = None):
    """Initializing the models attributes to a model_object variable.

    Args:
        model_name (str): The name of the model you wish to initialize. This should correspond to what your Model name.
        init (str): The initialization method you wish to use (Xavier support currently). Default is None.

    Returns:
        class: Object with the models class attributes
    """
    
    def xavier_init_weights(m):
        """
        Applies Xavier initialization to the weights of the given module.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    model_object = getattr(models, model_name)
    
    if init == "xavier":
        model_object.apply(xavier_init_weights)
    
    return model_object


def load_model(model_object, model_path: str, n_features: int, z_dim: int):
    """Loads the state dictionary of the trained model into a model variable. This variable is then used for passing
    data through the encoding and decoding functions.

    Args:
        model_object (object): Object with the models attributes
        model_path (str): Path to model
        n_features (int): Input dimension size
        z_dim (int): Latent space size

    Returns: nn.Module: Returns a model object with the attributes of the model class, with the selected state
    dictionary loaded into it.
    """
    device = get_device()
    model = model_object(n_features, z_dim)
    model.to(device)

    # Loading the state_dict into the model
    model.load_state_dict(
        torch.load(str(model_path), map_location=device), strict=False
    )
    return model


class Log1pScaler(BaseEstimator, TransformerMixin):
    """Log(1+x) transformer for positive-skewed HEP features"""
    def __init__(self):
        self.epsilon = 1e-8  # Small value to prevent log(0)
        
    def fit(self, X, y=None):
        if np.any(X + self.epsilon <= 0):
            raise ValueError("Data contains values <= 0 after epsilon addition")
        return self
    
    def transform(self, X):
        return np.log1p(X + self.epsilon)
    
    def inverse_transform(self, X):
        return np.expm1(X) - self.epsilon


class L2Normalizer(BaseEstimator, TransformerMixin):
    """L2 normalization per feature of data"""
    def __init__(self):
        self.norms = None
        
    def fit(self, X, y=None):
        self.norms = np.linalg.norm(X, axis=0)
        self.norms[self.norms == 0] = 1.0  # Prevent division by zero
        return self
    
    def transform(self, X):
        return X / self.norms
    
    def inverse_transform(self, X):
        return X * self.norms


class SinCosTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms an angle (in radians) into two features:
    [sin(angle), cos(angle)]. Inverse transformation uses arctan2.
    """
    def fit(self, X, y=None):
        # Nothing to learn
        return self
    
    def transform(self, X):
        # Ensure X is of shape (N,1)
        X = np.asarray(X).reshape(-1, 1)
        sin_part = np.sin(X)
        cos_part = np.cos(X)
        return np.hstack([sin_part, cos_part])
    
    def inverse_transform(self, X):
        if X.shape[1] != 2:
            raise ValueError("Expected input with 2 columns for inverse transformation.")
        sin_part = X[:, 0]
        cos_part = X[:, 1]
        angles = np.arctan2(sin_part, cos_part).reshape(-1, 1)
        return angles


class ChainedScaler(BaseEstimator, TransformerMixin):
    """
    Chains a list of scaler transformations.
    The transformation is applied sequentially (in the order provided)
    and the inverse transformation is applied in reverse order.
    """
    def __init__(self, scalers):
        self.scalers = scalers

    def fit(self, X, y=None):
        data = X
        for scaler in self.scalers:
            scaler.fit(data)
            data = scaler.transform(data)
        return self

    def transform(self, X):
        data = X
        for scaler in self.scalers:
            data = scaler.transform(data)
        return data

    def inverse_transform(self, X):
        data = X
        for scaler in reversed(self.scalers):
            data = scaler.inverse_transform(data)
        return data


def normalize_data(data, normalization_type):
    """
    Normalizes jet data for VAE-based anomaly detection.
    
    Args:
        data: 2D numpy array (n_jets, n_features)
        normalization_type: A string indicating the normalization method(s).
            It can be a single method or a chain of methods separated by '+'.
            Valid options include:
                'minmax'  - MinMaxScaler (scales features to [0,1])
                'standard'- StandardScaler (zero mean, unit variance)
                'robust'  - RobustScaler (less sensitive to outliers)
                'log'     - Log1pScaler (applies log1p transformation)
                'l2'      - L2Normalizer (scales each feature by its L2 norm)
                'power'   - PowerTransformer (using Yeo-Johnson)
                'quantile'- QuantileTransformer (transforms features to follow a normal or uniform distribution)
                'maxabs'  - MaxAbsScaler (scales each feature by its maximum absolute value)
                'sincos'  - SinCosTransformer (converts angles to sin/cos features)
            Example: 'log+standard' applies a log transformation followed by standard scaling.
    
    Returns:
        normalized_data: Transformed data array.
        scaler: Fitted scaler object (or chained scaler) for inverse transformations.
    """
    # Handle potential NaN/inf in HEP data
    if np.any(~np.isfinite(data)):
        raise ValueError("Input data contains NaN/infinite values")
    
    # Mapping from method names to corresponding scaler constructors.
    scaler_map = {
        'minmax': lambda: MinMaxScaler(feature_range=(0, 1)),
        'standard': lambda: StandardScaler(),
        'robust': lambda: RobustScaler(quantile_range=(5, 95)),  # Reduced outlier sensitivity
        'log': lambda: Log1pScaler(),
        'l2': lambda: L2Normalizer(),
        'power': lambda: PowerTransformer(method='yeo-johnson', standardize=True),
        'quantile': lambda: QuantileTransformer(output_distribution='normal'),
        'maxabs': lambda: MaxAbsScaler(),
        'sincos': lambda: SinCosTransformer(),
    }

    # Parse the chain of normalization methods.
    methods = normalization_type.split('+') if '+' in normalization_type else [normalization_type]
    
    scalers = []
    transformed_data = data.copy()
    
    for method in methods:
        method = method.strip().lower()
        if method not in scaler_map:
            raise ValueError(f"Unknown normalization method: {method}. "
                             "Valid options: " + ", ".join(scaler_map.keys()))
        scaler = scaler_map[method]()
        scaler.fit(transformed_data)
        transformed_data = scaler.transform(transformed_data)
        scalers.append(scaler)
    
    # If multiple scalers are used, return a chained scaler; otherwise the single scaler.
    if len(scalers) > 1:
        composite_scaler = ChainedScaler(scalers)
    else:
        composite_scaler = scalers[0]
    
    return transformed_data, composite_scaler


def invert_normalize_data(normalized_data, scaler):
    """
    Inverts a chained normalization transformation.

    This function accepts normalized data (for example, the output of a VAE's preprocessed input)
    and the scaler (or ChainedScaler) that was used to perform the forward transformation.
    It then returns the original data by calling the scaler's inverse_transform method.

    Args:
        normalized_data (np.ndarray): The transformed data array.
        scaler: The scaler object (or a ChainedScaler instance) used for the forward transformation,
                which must implement an `inverse_transform` method.

    Returns:
        np.ndarray: The data mapped back to its original scale.
    """
    if not hasattr(scaler, 'inverse_transform'):
        raise ValueError("The provided scaler object does not have an inverse_transform method.")
    return scaler.inverse_transform(normalized_data)


def load_sig_tensors(folder_path, keyword='sig_test'):
    """
    Searches through the specified folder for all '.pt' files containing the given keyword in their names.
    Categorizes these files based on the presence of 'jets', 'events', or 'constituents' in their filenames,
    loads them into PyTorch tensors, concatenates them along axis=0, and returns the resulting tensors.
    
    Args:
        folder_path (str): The path to the folder to search.
        keyword (str): The keyword to filter files ('bkg_train', 'bkg_test', or 'sig_test').
        
    Returns:
        tuple: A tuple containing three PyTorch tensors: (jets_tensor, events_tensor, constituents_tensor).
        
    Raises:
        ValueError: If any specific category ('jets', 'events', 'constituents') has no matching files.
                    The error message is:
                    "Required files not found. Please run the --mode convert_csv and prepare inputs before retrying."
    """
    if keyword not in ['bkg_train', 'bkg_test', 'sig_test']:
        raise ValueError("Invalid keyword. Please choose from 'bkg_train', 'bkg_test', or 'sig_test'.")

    # Initialize dictionaries to hold file lists for each category
    file_categories = {
        'jets': [],
        'events': [],
        'constituents': []
    }
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt') and keyword in filename:
            # Categorize files based on their names
            for category in file_categories:
                if category in filename:
                    file_categories[category].append(os.path.join(folder_path, filename))
    
    # Function to load and concatenate a list of .pt files along axis 0
    def load_and_concat(file_list):
        tensors = [torch.load(file) for file in file_list]
        return torch.cat(tensors, dim=0)
    
    # Load and concatenate tensors for each category
    result_tensors = {}
    for category, files in file_categories.items():
        if not files:
            raise ValueError("Required files not found. Please run the --mode convert_csv and prepare_inputs before retrying.")
        result_tensors[category] = load_and_concat(files)
    
    return result_tensors['events'], result_tensors['jets'], result_tensors['constituents']


def load_bkg_tensors(folder_path, keyword):
    """
    Searches through the specified folder for all '.pt' files whose names contain the specified
    keyword (e.g., 'bkg_train', 'bkg_test', or 'sig_test'). Files are then categorized by whether
    their filename contains one of the three substrings: 'jets', 'events', or 'constituents'.
    Additionally, each file must contain one of the generator names: 'herwig', 'pythia', or 'sherpa'.
    For each file, the tensor is loaded and a new feature is appended along the last dimension:
    - 0 for files containing 'herwig'
    - 1 for files containing 'pythia'
    - 2 for files containing 'sherpa'
    Finally, for each category the resulting tensors are concatenated along axis=0.
    
    Args:
        folder_path (str): The path to the folder to search.
        keyword (str): The keyword to filter files (e.g., 'bkg_train', 'bkg_test', or 'sig_test').
        
    Returns:
        tuple: A tuple of three PyTorch tensors: (jets_tensor, events_tensor, constituents_tensor)
               corresponding to the concatenated tensors for each category.
               
    Raises:
        ValueError: If any category does not have at least one file for each generator type.
                  The error message is:
                  "required files not found. please run the --mode convert_csv and prepare inputs before retrying"
    """
    # Check if the keyword is valid
    if keyword not in ['bkg_train', 'bkg_test', 'sig_test']:
        raise ValueError("Invalid keyword. Please choose from 'bkg_train', 'bkg_test', or 'sig_test'.")

    # Define the categories and generator subcategories.
    categories = ['jets', 'events', 'constituents']
    generators = {
        'herwig': 0,
        'pythia': 1,
        'sherpa': 2
    }
    
    # Initialize dictionary to store files per category and generator.
    file_categories = {cat: {gen: [] for gen in generators} for cat in categories}
    
    # Iterate over files in the folder.
    for filename in os.listdir(folder_path):
        # Only consider files ending with '.pt' that contain the specified keyword.
        if not filename.endswith('.pt'):
            continue
        if keyword not in filename:
            continue
        
        lower_filename = filename.lower()
        # Determine category based on substring in the filename.
        for cat in categories:
            if cat in lower_filename:
                # Determine generator type.
                for gen, gen_val in generators.items():
                    if gen in lower_filename:
                        full_path = os.path.join(folder_path, filename)
                        file_categories[cat][gen].append((full_path, gen_val))
                # Note: if a file contains multiple generator substrings (unlikely), it will be added
                # to all matching generator groups.
    
    # For each category, ensure that each generator type has at least one file.
    for cat in categories:
        for gen in generators:
            if len(file_categories[cat][gen]) == 0:
                raise ValueError("Required files not found. Please run the --mode convert_csv and prepare inputs before retrying.")
    
    # For each file, load its tensor and append the generator feature.
    def load_and_augment(file_info):
        """
        Given a tuple (file_path, generator_value), load the tensor and append a new feature column
        with the constant generator_value along the last dimension.
        Works for both 2D and 3D tensors.
        """
        file_path, gen_val = file_info
        tensor = torch.load(file_path)
        # Create a constant tensor with the same device and dtype as tensor.
        if tensor.dim() == 2:
            # For a 2D tensor of shape (m, n), create a (m, 1) tensor.
            constant_feature = torch.full((tensor.size(0), 1), gen_val, dtype=tensor.dtype, device=tensor.device)
            augmented = torch.cat([tensor, constant_feature], dim=1)
        elif tensor.dim() == 3:
            # For a 3D tensor of shape (m, p, n), create a (m, p, 1) tensor.
            constant_feature = torch.full((tensor.size(0), tensor.size(1), 1), gen_val, dtype=tensor.dtype, device=tensor.device)
            augmented = torch.cat([tensor, constant_feature], dim=-2)
        else:
            raise ValueError("Tensor from {} has unsupported dimensions: {}".format(file_path, tensor.dim()))
        return augmented

    # For each category, load the tensors for each generator, augment them, and then concatenate.
    concatenated = {}
    for cat in categories:
        cat_tensors = []
        for gen in generators:
            # Get the list of file infos (tuples) for this generator.
            file_list = file_categories[cat][gen]
            # For each file, load and augment.
            for file_info in file_list:
                cat_tensors.append(load_and_augment(file_info))
        # Before concatenation, we want to split the data into a multiple of the sample count
        # (here we simply concatenate along axis=0).
        concatenated[cat] = torch.cat(cat_tensors, dim=0)
    
    return concatenated['events'], concatenated['jets'], concatenated['constituents']


def train_val_split(tensor, train_ratio):
    """
    Splits a tensor into training and validation sets based on the specified train_ratio.
    The split is done by sampling indices randomly ensuring that the data is shuffled.
    
    Args:
        tensor (torch.Tensor): The input tensor to be split.
        train_ratio (float): Proportion of data to be used for training (e.g., 0.8 for 80% training data).
        
    Returns:
        tuple: A tuple containing two tensors:
            - train_tensor: Tensor containing the training data.
            - val_tensor: Tensor containing the validation data.
            
    Raises:
        ValueError: If train_ratio is not between 0 and 1.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be a float between 0 and 1.")
    
    # Set the random seed for reproducibility.
    torch.manual_seed(42)

    # Determine the split sizes
    total_size = tensor.size(0)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Generate a random permutation of indices.
    indices = torch.randperm(total_size)

    # Split the indices into train and validation indices.
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Use these indices to index into your tensor.
    train_tensor = tensor[train_indices]
    val_tensor = tensor[val_indices]
    
    return train_tensor, val_tensor

import torch


def get_optimizer(optimizer_name, parameters, lr):
    """
    Returns a PyTorch optimizer configured with optimal arguments for training a large VAE.
    
    Args:
        optimizer_name (str): One of "adam", "adamw", "rmsprop", "sgd", "radam", "adagrad".
        parameters (iterable): The parameters (or parameter groups) of your model.
        lr (float): The learning rate for the optimizer.
        
    Returns:
        torch.optim.Optimizer: An instantiated optimizer with specified hyperparameters.
        
    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
    opt = optimizer_name.lower()

    if opt == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=(0.9, 0.999),  # Default values
            eps=1e-8,
            weight_decay=0  # Set to a small value like 1e-5 if regularization is needed
        )
    elif opt == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2  # L2 regularization
        )
    elif opt == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=config.lr,
            alpha=0.99,  # Smoothing constant
            eps=1e-8,
            weight_decay=1e-2,  # L2 regularization
            momentum=0.9  # Set to a value like 0.9 if momentum is desired
        )
    elif opt == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=0.9,  # Momentum term
            weight_decay=0,  # Set to a small value like 1e-5 if regularization is needed
            nesterov=True  # Set to True if Nesterov momentum is desired
        )
    elif opt == "radam":
        return torch.optim.RAdam(
            parameters,
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0
        )
    elif opt == "adagrad":
        return torch.optim.Adagrad(
            parameters,
            lr=config.lr,
            lr_decay=0,  # Learning rate decay over each update
            weight_decay=0,
            initial_accumulator_value=0,  # Starting value for the accumulators
            eps=1e-10
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def call_forward(model, inputs):
    """
    Calls the `forward` method of the given object.
    If the return value is not a tuple, packs it into a tuple.
    
    Args:
        model: An object that has a `forward` method.
        inputs: The input data to pass to the model.
    
    Returns:
        A tuple containing the result(s) of the `forward` method.
    """
    # Call the forward method
    result = model(inputs)
    
    # Ensure the result is a tuple
    if isinstance(result, tuple):
        return result
    else:
        return (result,)

