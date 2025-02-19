[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14888114.svg)](https://doi.org/10.5281/zenodo.14888114)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![GitHub version](https://badge.fury.io/gh/PRAkTIKal24%2FBEAD.svg)](https://badge.fury.io/gh/PRAkTIKal24%2FBEAD)
[![example event workflow](https://github.com/PRAkTIKal24/BEAD/actions/workflows/docs.yaml/badge.svg?event=push)](https://github.com/PRAkTIKal24/BEAD/actions)

# BEAD
**Background Enrichment for Anomaly Detection**
|| *{Background Enrichis pour (la Détection d'Anomalies) Anomalie Détection}*



BEAD is a Python package that uses deep learning based methods for anomaly detection in HEP data for new physics searches. BEAD has been designed with modularity in mind, to enable usage of various unsupervised latent variable models for any task.

BEAD has five main running modes:

- Data handling: Deals with handling file types, conversions between them and pre-processing the data to feed as inputs to the DL models.

- Training: Train your model to learn implicit representations of your background data that may come from multiple sources/generators to get a single, encriched latent representation of it.

- Inference: Using a model trained on an enriched background, feed in samples you want to detect anomalies in and watch the magic happen.

- Plotting: After running Inference, or Training you can generate plots similar towhat is shown in the paper. These include performance plots as well as different visualizations of the learned data.

- Diagnostics: Enabling this mode allows running profilers that measure a host of metrics connected to the usage of the compute node you run on to help you optimize the code if needed(using CPU-GPU metrics).

For more information, follow the install instructions below go to the directory `./BEAD/bead/` and then run the command ```poetry run bead -h``` to get detailed instructions on running the package and the available customizations



# Installation
1. Poetry: BEAD is managed by the poetry package manager - this simplifies the task of creating an environment, installing the right dependencies, version incompatibilities etc. So first start with installing poetry according to the instructions given [here](https://python-poetry.org/docs/#installation)
2. After installing poetry, clone this repository to your working directory.
3. Enter the `BEAD/bead/` directory using `cd bead`
4. You are now ready to start running the package! As a first step try the following command:
```poetry run bead -h```  
This should bring up the help window that explains all the various running modes of bead.
5. Start with creating a new workspace and project like so:
```poetry run bead -m new_project -p <WORKSPACE_NAME> <PROJECT_NAME>```
This will setup all the required directories inside `BEAD/bead/workspaces/`. Remember to use a different workspace everytime you want to modify your input data, since all the projects inside a given workspace share and overwrite the input data. If you want to use the same input data but change something else in the pipeline (for eg. different config options such as `model_name`, `loss_function` etc.), use the same `workspace_name`, but create a new project with a different `'project_name'`. On doing this, your data will already be ready from the previous project in that workspace so you can skip directly to the subsequent steps.
6. After creating a new workspace, it is essential to move the `*_input_data.csv` files to the `BEAD/bead/workspaces/WORKSPACE_NAME/data/` directory
7. After making sure the input files are in the right location, you can run the subsequent operation modes of BEAD.
