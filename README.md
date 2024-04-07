# Unsupervised Coreset Selection

An implementation of core-set selection for image segmentation tasks.

## Install dependecies

```bash
pip install -r requirements.txt
```

## Usage

Run the following script to start training SimCLR and export coreset dataloader to disk:

```bash
python coreset-select.py
```

> [!Note]
> Your custom dataloader should implement the appropriate augmentation pipeline discussed in [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).
> This script also assumes you have the appropriate CUDA dependencies installed for training.

## References

Ju, J. et al. (2021) Extending contrastive learning to unsupervised Coreset selection, arXiv.org. Available at: https://arxiv.org/abs/2103.03574 (Accessed: 01 February 2024). 
