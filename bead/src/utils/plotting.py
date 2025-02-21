# Description: This file contains a function to generate plots for training epoch loss data and loss component data for train, val and test sets based on what files exist in the output directory.
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.rich import tqdm


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
        plt.plot(epochs, train_epoch_loss, label='Train Loss', marker='o')
        plt.plot(epochs, val_epoch_loss, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(config.project_name)
        plt.legend()
        plt.tight_layout()
        
        # Save plot as PDF
        plt.savefig(os.path.join(save_dir, "train_metrics.pdf"))
        plt.close()
    else:
        raise FileNotFoundError("Epoch loss data files not found. Make sure to run the train mode first.")
    
    # --------- Plot Loss Components ---------
    # Define the prefixes and categories of interest
    prefixes = ['loss', 'reco', 'kl', 'emd', 'l1', 'l2']
    categories = ['train', 'test', 'val']
    
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
                
                plt.plot(epochs, avg_losses, label=label, marker='o')
            
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
            raise FileNotFoundError(f"No loss component data files found for {cat} set. Make sure to run the appropriate {cat} mode first.")

    if verbose:
        print("Loss plots generated successfully and saved to: ", save_dir)
