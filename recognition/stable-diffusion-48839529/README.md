# Stable Diffusion trained on ADNI Brain MRI Dataset

This project aims to train a generative image model to produce synthetic BRAIN-MRI images without simply memorizing the training dataset. This project emphasises anonymity of the patients for which the training dataset was made and so a major portion of this project is detection of memorization of training data by the model.

# The Algorithm: Stable Diffusion

Stable Diffusion is an averarching term covering many different architectures. For this particular task I trained a Latent Diffusion model. This model consists of two major components, the VAE and the Denoising UNET. [(Rombach et al., 2022)](https://arxiv.org/pdf/2112.10752)

![Latent Diffusion Architecture](./resources/diffusion_architecture.png)

The above architecture diagram was taken directly from the paper. For this implementation we will disregard the transformer-based conditioning. Below is the architecture diagram of the VQVAE taken from the referenced paper [(Esser et al., CVPR 2021)](https://arxiv.org/pdf/2012.09841)

![VQVAE Architecture Diagram](./resources/image.png)

The Denoising UNET Architecture was adapted from assignment 2 with the addition of some Attention layers within the bottneck portion of the model which is consistent with modern UNET implementations.

![UNET Architecture Diagram](image.png)
[(Ronneberger et al, 2015)](https://arxiv.org/pdf/1505.04597)

The Stable Diffusion model works by first encoding the image into latent space via a pretrained VQVAE. The UNET is then trained to predict the added noise to an image for a certain step. For this implementation I utilized a linear nosing schedule. Once the UNET and VQVAE are finished training, the encoder can be disgarded and only the unet and decoder utilized to generate images. This happens by first sampling from the multivariate gaussian, iteratively predicting the added noise, and then numerically integrating along the denosing trajectory for a predefined number of timesteps to recover the image. 

# Preprocessing
The dataset used was the ADNI Brain MRI dataset. The dataloader supplied in dataset.py incorporates basic image rescaling and greyscale conversion for data preprocessing as this was found to be sufficient.

# Proposal and Feasibility Review

## User Need, Scope & Acceptance Criteria

The proposed research dilemma details a problem in curation of clinical datasets for open research. These datasets are typically sparse due to problems in patient confidentiality which limit the available, legally accessible data. A potential solution to this problem is to generate synthetic training examples. However, certain methods such as VAE typically generate poor and/or blurry images, leaving room for a better model that can generate high fidelity images whilst respecting patent confidentiality. 

The typical users of such a solution are medical students and researchers. These uses require the following:
- Anatomical representations are not blurry 
- The generated images do not hallucinate physiologically impossible tissue junctions or non-existant pathologies
- The model doesn't memorize patient scans from the dataset verbatim

To judge the perfomance of the solution against the above user requirements, the following acceptance criteria has been decided:
- A large number of samples will be generated and used to evaluate:
    - Visually inspect each image and compare it to the patient dataset. This will be a visual comparison to assess blurriness and physiological accuracy
    - For determining whether the generative model memorizes training data, the sample will be evaluated against the training and test datasets using the Mann–Whitney U test. This is a test for whether two datasets come from the same distribution. This test will generate p-values for both datasets and if the likelihood is higher for the test set than the training set then we can say the generative model likely didn't memorize the training data.

## Model Choice & Course Concepts

For this project I have chosen stable diffusion model as it has become (close to) the SOTA for generating crisp images in various domains. To successfully complete this task the following course concepts will be used:
- VQVAE and diffusion theory from lectures
- Best practices when writing code that runs on a HPC cluster

## Preliminary Feasibility Evidence

I have created and trained a VQVAE model on Rangpur already which has achieved good results on the ADNI dataset. 

![Preliminary VQVAE Loss Curve](./resources/preliminary_vqvae_loss_curve.png)

## Risks, Budget and Fallback

Risks
- The diffusion model proves too difficult or time consuming to train
    - *Fallback*: Since the diffusion model has A VQ-VAE as an integral component, worst case I can submit the VQ-VAE as my project instead
- Rangpur cluster gets too busy to be able to train the model
    - *Mitigation*: I have budgeted significant enough time to this project to allow me to train the model earlier during the project period. This should give me enough time to avoid the rush.

Budget:
The latent diffusion model gave significant speed boosts to the generative modeling training community. The paper notes that training on imagenet using an A100 GPU took roughly 6.66hrs/epoch (VQVAE) for which they trained the model for roughly 2M steps taking around 1-2weeks of gpu training time. Considering this data and the relavtive size of ADNI (10,400 images) versus imagenet (1,281,167 images 1,000 classes) the time to train should be significantly smaller 10400/1281167 = 0.8% ~4hrs (conservatively) for training the diffusion portion of the model and from abalation ~3hrs to train the VQVAE portion. 

Total estimated training time for a full run ~7hrs.

# Requirements
This project was created using python version (3.14<=). The required packages are:
- Matplotlib
- Pytorch 
- Argparse/pathlib

# The Open Research Dilemma

![Sharpness Successful Sample](./resources/sharplow01.png)
![Sharpness Poor Sample](./resources/sharplow02.png)