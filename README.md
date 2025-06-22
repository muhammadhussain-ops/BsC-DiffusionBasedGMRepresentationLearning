# Gaussian Mixture Model Representations Based on Stochastic Differential Equations

**B.Sc. Thesis Project**  
**Muhammad Hussain Altakmaji (s225785)**  
Department of Applied Mathematics and Computer Science, DTU Compute, Technical University of Denmark  
June 2025

---

## Project Description

This repository contains code and experiments from my bachelor’s thesis, where I investigate how to construct latent representations governed by a Gaussian Mixture Model (GMM) prior within a stochastic differential equation (SDE)-based diffusion model. I replicate and extend “Diffusion Based Representation Learning” by replacing the standard Gaussian prior with a GMM prior, and introduce the novel **JGMVDRL objective**, which combines the strengths of GMVAE and SDE-based diffusion models.

---

## Key Contributions

- **GMVAE Construction**  
  Implementation of a Gaussian Mixture Variational Autoencoder that encodes data into a latent space with a GMM prior.  
- **Diffusion-Based Representation Learning**  
  Replication of both JDRL and VDRL formulations on MNIST using a variance-preserving (VP) SDE.  
- **JGMVDRL Objective**  
  A new loss function that integrates the GMM prior from GMVAE with SDE-based diffusion to create more structured latent spaces.  
- **Experimental Results**  
  Demonstration of clear, class-related clusters in 2D latent spaces under different sampling schedules.

---

## Installation

```bash
git clone https://github.com/<your-username>/BsC-DiffusionBasedGMRepresentationLearning.git
cd BsC-DiffusionBasedGMRepresentationLearning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

