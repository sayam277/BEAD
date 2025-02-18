[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14888114.svg)](https://doi.org/10.5281/zenodo.14888114)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![GitHub version](https://badge.fury.io/gh/PRAkTIKal24%2FBEAD.svg)](https://badge.fury.io/gh/PRAkTIKal24%2FBEAD)
![example event workflow](https://github.com/PRAkTIKal24/BEAD/actions/workflows/docs.yaml/badge.svg?event=push)

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
