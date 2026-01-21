# Finite element simulations of brain ventricular cerebrospinal fluid (CSF) flow

This repository contains software for finite element simulations of CSF flow.
The software implements finite element methods with the FEniCS Python interface DOLFINx.

## Getting started
Start by cloning this repository and entering the root directory.
Install the required packages using `conda` with
```
conda env create -f environment.yml
```
This installs the packages in a `conda` environment named
`lagrangian-csf-env`, which can be activated with the command
```
conda activate lagrangian-csf-env
```
