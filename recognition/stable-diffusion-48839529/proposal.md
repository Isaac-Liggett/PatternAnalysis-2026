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

![Preliminary VQVAE Loss Curve](./preliminary_vqvae_loss_curve.png)

## Risks, Budget and Fallback

Risks
- The diffusion model proves too difficult or time consuming to train
    - *Fallback*: Since the diffusion model has A VQ-VAE as an integral component, worst case I can submit the VQ-VAE as my project instead
- Rangpur cluster gets too busy to be able to train the model
    - *Mitigation*: I have budgeted significant enough time to this project to allow me to train the model earlier during the project period. This should give me enough time to avoid the rush.

Budget:
The latent diffusion model gave significant speed boosts to the generative modeling training community. The paper notes that training on imagenet using an A100 GPU took roughly 6.66hrs/epoch (VQVAE) for which they trained the model for roughly 2M steps taking around 1-2weeks of gpu training time. Considering this data and the relavtive size of ADNI (10,400 images) versus imagenet (1,281,167 images 1,000 classes) the time to train should be significantly smaller 10400/1281167 = 0.8% ~4hrs (conservatively) for training the diffusion portion of the model and from abalation ~3hrs to train the VQVAE portion. 

Total estimated training time for a full run ~7hrs.
