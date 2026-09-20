# SparSSL: Exploiting Sparsity for MNIST Classification via Self-Supervised Pretraining

EE4C12 (Machine Learning for Electrical Engineering) final project, TU Delft.

## Overview

Classification of the MNIST dataset framed around measurement-limited settings:
images are reconstructed from a subset of *m* measurements (random masking /
compressed sensing), and classification error is studied as a function of *m*.

Beyond the standard supervised pipeline, we test whether self-supervised
pretraining (masked-reconstruction pretext task via an encoder-decoder MLP)
improves classification accuracy under measurement scarcity, compared to
training a classifier from scratch at the same *m*.

## Models

- Classical baseline: logistic regression / SVM on masked pixel vectors
- Supervised DNN: MLP trained end-to-end on the classification task
- SSL: MLP encoder-decoder pretrained on reconstruction (unsupervised, full
  60k set), classification head attached to the frozen/fine-tuned encoder

## Setup

\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`

## Authors

<your name> (<student number>), <colleague's name> (<student number>)

## License

MIT — see LICENSE.
