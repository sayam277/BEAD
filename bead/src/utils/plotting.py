# Description: This file contains a function to generate plots for training epoch loss data and loss component data for train, val and test sets based on what files exist in the output directory.
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.rich import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import trimap


def plot_losses(output_dir, save_dir, config, verbose: bool = False):
    """
    Generates plots for training epoch loss data and loss component data for train, val and test sets based on what files exist in the output directory.

    Parameters:
        output_dir (str): The path to the directory containing output .npy files.
        save_dir (str): The path to the directory where the plots will be saved.
        config: A config object that defines user choices.
        verbose (bool): If True, print progress messages.

    """
    if verbose:
        print("Making Loss Plots...")
    # --------- Plot Train & Validation Epoch Losses ---------
    train_loss_file = os.path.join(output_dir, "train_epoch_loss_data.npy")
    val_loss_file = os.path.join(output_dir, "val_epoch_loss_data.npy")

    if os.path.exists(train_loss_file) and os.path.exists(val_loss_file):
        # Load epoch-wise loss arrays (assumed 1D arrays with one value per epoch)
        train_epoch_loss = np.load(train_loss_file)
        val_epoch_loss = np.load(val_loss_file)

        epochs = np.arange(1, len(train_epoch_loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_epoch_loss, label="Train Loss", marker="o")
        plt.plot(epochs, val_epoch_loss, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(config.project_name)
        plt.legend()
        plt.tight_layout()

        # Save plot as PDF
        plt.savefig(os.path.join(save_dir, "train_metrics.pdf"))
        plt.close()
    else:
        raise FileNotFoundError(
            "Epoch loss data files not found. Make sure to run the train mode first."
        )

    # --------- Plot Loss Components ---------
    # Define the prefixes and categories of interest
    prefixes = ["loss", "reco", "kl", "emd", "l1", "l2"]
    categories = ["train", "test", "val"]

    # Iterate over each category
    for cat in categories:
        files_for_cat = []
        # For each prefix, build the file path and check if it exists.
        for prefix in prefixes:
            file_path = os.path.join(output_dir, f"{prefix}_{cat}.npy")
            if os.path.exists(file_path):
                files_for_cat.append(file_path)

        # If any files are found for the current category, process and plot them.
        if files_for_cat:
            plt.figure(figsize=(8, 6))

            for file in files_for_cat:
                data = np.load(file)
                num_epochs = config.epochs
                total_length = len(data)

                # Determine the number of events per epoch.
                # Assumes that total_length divides evenly by config.epochs.
                events_per_epoch = total_length // num_epochs

                avg_losses = []
                for epoch in range(num_epochs):
                    start_idx = epoch * events_per_epoch
                    end_idx = start_idx + events_per_epoch
                    epoch_data = data[start_idx:end_idx]
                    avg_losses.append(np.mean(epoch_data))
                avg_losses = np.array(avg_losses)

                epochs = np.arange(1, num_epochs + 1)
                # Use the file prefix (without the _category.npy) as the label.
                base_name = os.path.basename(file)
                label = os.path.splitext(base_name)[0]

                plt.plot(epochs, avg_losses, label=label, marker="o")

            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title(config.project_name)
            plt.legend()
            plt.tight_layout()

            # Save the plot to a file with category in its name.
            save_filename = os.path.join(save_dir, f"loss_components_{cat}.pdf")
            plt.savefig(save_filename)
            plt.close()

        else:
            raise FileNotFoundError(
                f"No loss component data files found for {cat} set. Make sure to run the appropriate {cat} mode first."
            )

    if verbose:
        print("Loss plots generated successfully and saved to: ", save_dir)


def plot_latent_variables(config, paths, verbose=False):
    """
    This function loads the latent representations (z0 and zk) and corresponding labels,
    checks for consistency between the number of background samples and gen_labels,
    reduces the dimensions using either PCA, t-SNE or TriMap (with an initial PCA reduction to 50 dims if needed),
    plots the embeddings for background (colored according to gen_labels) and signal events,
    and saves the plots as PDF files.

    Parameters:
      config (dataClass): An object with user defined config options
      paths (dictionary): Dictionary of common paths used in the pipeline
      verbose: If True, prints additional debug information.

    """
    # Construct file paths
    label_file = os.path.join(
        paths["output_path"], "results", config.input_level + "_label.npy"
    )
    gen_label_file = os.path.join(
        paths["data_path"],
        config.file_type,
        "tensors",
        "processed",
        "gen_label_" + config.input_level + ".npy",
    )
    z0_file = os.path.join(paths["output_path"], "results", "z0_data.npy")
    zk_file = os.path.join(paths["output_path"], "results", "zk_data.npy")

    # Load data
    labels = np.load(label_file)
    gen_labels = np.load(gen_label_file)
    z0 = np.load(z0_file)
    zk = np.load(zk_file)

    # Check consistency: count of zeros in labels must equal the size of gen_labels
    n_background = np.sum(labels == 0)
    if len(gen_labels) != n_background:
        raise ValueError(
            "Mismatch: gen_label size ({}) does not match number of background samples in label.npy ({})".format(
                len(gen_labels), n_background
            )
        )
    if verbose:
        print(
            "Loaded {} total samples: {} background and {} signal samples.".format(
                len(labels), n_background, len(labels) - n_background
            )
        )
        print("Loaded gen_labels with {} entries.".format(len(gen_labels)))

    def reduce_dim(data):
        """
        Reduce the dimension of the input data to 2D for plotting.
        If config.latent_space_size > 50, first reduce to 50 components via PCA.
        Then apply the method specified in config.latent_space_plot_style.
        Returns:
          reduced: data reduced to 2D.
          method_used: string indicating the reduction method used.
        """
        # If original latent space is high-dimensional, first reduce to 50 dimensions.
        if config.latent_space_size > 50:
            if verbose:
                print(
                    "Applying PCA to reduce latent space from {} to 50 dimensions...".format(
                        config.latent_space_size
                    )
                )
            data = PCA(n_components=50).fit_transform(data)

        style = config.latent_space_plot_style.lower()
        if style == "pca":
            if verbose:
                print("Reducing to 2 dimensions using PCA.")
            reduced = PCA(n_components=2).fit_transform(data)
            method_used = "pca"
        elif style == "tsne":
            if verbose:
                print("Reducing to 2 dimensions using t-SNE.")
            reduced = TSNE(n_components=2, random_state=42).fit_transform(data)
            method_used = "tsne"
        elif style == "trimap":
            if verbose:
                print("Reducing to 2 dimensions using TriMap.")
            reduced = trimap.TRIMAP(n_dims=2).fit_transform(data)
            method_used = "trimap"
        else:
            raise ValueError(
                "Invalid latent_space_plot_style: {}. Must be 'pca', 'tsne', or 'trimap'.".format(
                    config.latent_space_plot_style
                )
            )
        return reduced, method_used

    # Create a color mapping for the events.
    # Background events: first n_background indices, with gen_labels: 0->green, 1->blue, 2->yellow.
    # Signal events: all remaining indices, colored red.
    colors = []
    for i in range(n_background):
        if gen_labels[i] == 0:
            colors.append("green")
        elif gen_labels[i] == 1:
            colors.append("blue")
        elif gen_labels[i] == 2:
            colors.append("yellow")
        else:
            # If an unexpected gen_label is encountered, default to black.
            colors.append("black")
    n_signal = len(labels) - n_background
    colors.extend(["red"] * n_signal)

    # Loop over the two latent representations and produce plots.
    for latent_data, suffix in [(z0, "_z0"), (zk, "_zk")]:
        if verbose:
            print("Processing latent data for '{}'...".format(suffix))
        reduced_data, method_used = reduce_dim(latent_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=colors,
            alpha=0.7,
            edgecolors="w",
            s=60,
        )
        plt.title(
            "{} embedding using {}".format(suffix[1:].upper(), method_used.upper())
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        # Create custom legend
        import matplotlib.patches as mpatches

        legend_handles = [
            mpatches.Patch(color="green", label="Herwig (gen_label 0)"),
            mpatches.Patch(color="blue", label="Pythia (gen_label 1)"),
            mpatches.Patch(color="yellow", label="Sherpa (gen_label 2)"),
            mpatches.Patch(color="red", label="Signal"),
        ]
        plt.legend(handles=legend_handles)

        # Save the plot as a PDF
        save_path = os.path.join(
            paths["output_path"],
            "plots",
            "latent_space",
            "{}{}.pdf".format(config.project_name, suffix),
        )
        plt.savefig(save_path, format="pdf")
        if verbose:
            print("Saved plot to '{}'".format(save_path))
        plt.close()


def plot_mu_logvar(config, paths, verbose=False):
    """
    This function loads the latent distribution parameters (mu and logvar) along with label information,
    checks consistency between background samples and gen_labels, and then creates:
      1. A scatter plot of the latent means (mu) reduced to 2D using the specified method (PCA, t-SNE, or TriMap).
      2. A histogram of an uncertainty metric computed from logvar.

    The plots are saved as PDFs in the same directory with filenames based on config.project_name.

    Parameters:
      config (dataClass): An object with user defined config options
      paths (dictionary): Dictionary of common paths used in the pipeline
      verbose: If True, prints additional debug information.

    """
    # Construct file paths
    mu_file = os.path.join(paths["output_path"], "results", "mu_data.npy")
    logvar_file = os.path.join(paths["output_path"], "results", "logvar_data.npy")
    label_file = os.path.join(
        paths["output_path"], "results", config.input_level + "_label.npy"
    )
    gen_label_file = os.path.join(
        paths["data_path"],
        config.file_type,
        "tensors",
        "processed",
        "gen_label_" + config.input_level + ".npy",
    )

    # Load data
    mu = np.load(mu_file)
    logvar = np.load(logvar_file)
    labels = np.load(label_file)
    gen_labels = np.load(gen_label_file)

    # Check that the number of background samples (zeros in label) matches the length of gen_labels
    n_background = np.sum(labels == 0)
    if len(gen_labels) != n_background:
        raise ValueError(
            "Mismatch: gen_label size ({}) does not match number of background samples in label.npy ({})".format(
                len(gen_labels), n_background
            )
        )
    if verbose:
        print(
            "Loaded {} total samples: {} background and {} signal samples.".format(
                len(labels), n_background, len(labels) - n_background
            )
        )
        print(
            "Loaded mu_data with shape {} and logvar_data with shape {}.".format(
                mu.shape, logvar.shape
            )
        )

    def reduce_dim(data):
        """
        Reduce input data to 2D using the specified method.
        If config.latent_space_size > 50, first reduce data to 50 dimensions using PCA.

        Returns:
          reduced: data reduced to 2D.
          method_used: string indicating the reduction method.
        """
        if config.latent_space_size > 50:
            if verbose:
                print(
                    "Performing initial PCA to reduce from {} to 50 dimensions.".format(
                        config.latent_space_size
                    )
                )
            data = PCA(n_components=50).fit_transform(data)

        style = config.latent_space_plot_style.lower()
        if style == "pca":
            if verbose:
                print("Reducing to 2 dimensions using PCA.")
            reduced = PCA(n_components=2).fit_transform(data)
            method_used = "pca"
        elif style == "tsne":
            if verbose:
                print("Reducing to 2 dimensions using t-SNE.")
            reduced = TSNE(n_components=2, random_state=42).fit_transform(data)
            method_used = "tsne"
        elif style == "trimap":
            if verbose:
                print("Reducing to 2 dimensions using TriMap.")
            reduced = trimap.TRIMAP(n_dims=2).fit_transform(data)
            method_used = "trimap"
        else:
            raise ValueError(
                "Invalid latent_space_plot_style: {}. Choose from 'pca', 'tsne', or 'trimap'.".format(
                    config.latent_space_plot_style
                )
            )
        return reduced, method_used

    # Define color mapping for background events based on gen_labels and red for signal events.
    colors = []
    for i in range(n_background):
        if gen_labels[i] == 0:
            colors.append("green")
        elif gen_labels[i] == 1:
            colors.append("blue")
        elif gen_labels[i] == 2:
            colors.append("yellow")
        else:
            colors.append("black")
    n_signal = len(labels) - n_background
    colors.extend(["red"] * n_signal)

    # -------------------------------
    # Plot 1: Latent Means (mu)
    # -------------------------------
    if verbose:
        print("Reducing mu data to 2D for scatter plot...")
    reduced_mu, method_used = reduce_dim(mu)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_mu[:, 0], reduced_mu[:, 1], c=colors, alpha=0.7, edgecolors="w", s=60
    )
    plt.title("Latent Means (mu) embedding using {}".format(method_used.upper()))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Create custom legend
    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(color="green", label="Herwig (gen_label 0)"),
        mpatches.Patch(color="blue", label="Pythia (gen_label 1)"),
        mpatches.Patch(color="yellow", label="Sherpa (gen_label 2)"),
        mpatches.Patch(color="red", label="Signal"),
    ]
    plt.legend(handles=legend_handles)

    # Save the mu plot as a PDF
    mu_save_path = os.path.join(
        paths["output_path"],
        "plots",
        "latent_space",
        "{}_mu.pdf".format(config.project_name),
    )
    plt.savefig(mu_save_path, format="pdf")
    if verbose:
        print("Saved latent means plot to '{}'".format(mu_save_path))
    plt.close()

    # -------------------------------
    # Plot 2: Uncertainty from logvar
    # -------------------------------
    # Compute a per-sample uncertainty measure. One common metric is the mean standard deviation:
    # sigma = exp(0.5 * logvar) and then uncertainty = mean(sigma) across latent dimensions.
    sigma = np.exp(0.5 * logvar)
    uncertainty = np.mean(sigma, axis=1)

    # Separate uncertainty for background and signal samples.
    uncertainty_bkg = uncertainty[:n_background]
    uncertainty_sig = uncertainty[n_background:]

    # For background, further split based on generator (gen_label)
    uncertainty_herwig = uncertainty_bkg[gen_labels == 0]
    uncertainty_pythia = uncertainty_bkg[gen_labels == 1]
    uncertainty_sherpa = uncertainty_bkg[gen_labels == 2]

    plt.figure(figsize=(8, 6))
    bins = 30
    plt.hist(
        uncertainty_herwig,
        bins=bins,
        color="green",
        alpha=0.6,
        label="Herwig (gen_label 0)",
    )
    plt.hist(
        uncertainty_pythia,
        bins=bins,
        color="blue",
        alpha=0.6,
        label="Pythia (gen_label 1)",
    )
    plt.hist(
        uncertainty_sherpa,
        bins=bins,
        color="yellow",
        alpha=0.6,
        label="Sherpa (gen_label 2)",
    )
    plt.hist(uncertainty_sig, bins=bins, color="red", alpha=0.6, label="Signal")
    plt.xlabel("Mean Standard Deviation (uncertainty)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Latent Uncertainty")
    plt.legend()

    # Save the uncertainty plot as a PDF
    uncertainty_save_path = os.path.join(
        paths["output_path"],
        "plots",
        "latent_space",
        "{}_uncertainty.pdf".format(config.project_name),
    )
    plt.savefig(uncertainty_save_path, format="pdf")
    if verbose:
        print("Saved uncertainty plot to '{}'".format(uncertainty_save_path))
    plt.close()


def plot_roc_curve(config, paths, verbose: bool = False):
    """
    Generates and saves ROC curves for available loss component files.

    Parameters:
      config (dataClass): An object with user defined config options
      paths (dictionary): Dictionary of common paths used in the pipeline
      verbose: If True, prints additional debug information.

    """
    # Load ground truth binary labels from 'label.npy'
    label_path = os.path.join(
        paths["output_path"], "results", config.input_level + "_label.npy"
    )
    output_dir = os.path.join(paths["output_path"], "results")
    ground_truth = np.load(label_path)

    # Ensure ground_truth is a 1D array
    if ground_truth.ndim != 1:
        raise ValueError("Ground truth labels must be a 1D array.")

    # Define the loss component prefixes to search for.
    loss_components = ["loss", "reco", "kl", "emd", "l1", "l2"]

    # Iterate over each loss component and generate ROC curve.
    plt.figure(figsize=(8, 6))

    for component in loss_components:
        file_path = os.path.join(output_dir, f"{component}_test.npy")

        if not os.path.exists(file_path):
            continue  # Skip if the file does not exist

        # Load loss scores
        data = np.load(file_path)

        # Ensure that data is a 1D array (flatten if necessary).
        if data.ndim > 1:
            data = data.flatten()

        # Check if the length of data matches the length of ground_truth
        if len(data) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {file_path} has {len(data)} entries; "
                f"ground truth has {len(ground_truth)} entries."
            )

        # Compute ROC curve and AUC.
        fpr, tpr, thresholds = roc_curve(ground_truth, data)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve.
        plt.plot(fpr, tpr, label=f"{component.capitalize()}) AUC = {roc_auc:.2f}", lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {config.project_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save the plot as a PDF file.
    save_filename = os.path.join(paths["output_path"], "plots", "loss", "roc.pdf")
    plt.savefig(save_filename)
    plt.close()
