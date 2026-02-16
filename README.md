# miniVLA
Small-scale proof of concept vision-language-action (VLA) model based on Keivalya Pandya's tutorial

Model components
* Text Encoder (encodes language instruction, implemented here as a simple GRU)
* Image Encoder (encodes image inputs from sensing, implemented here as a simple CNN)
* State Encoder (encodes robot state vector, here as an MLP i.e. simple FC network)
* Fusion module (combines encoded inputs via concatenation)
* Diffusion Head: Implements DPPM (Denoising Diffusion Probabilistic Model), performing denoising by predicting the noise added at each time step
* Diffusion Policy: Combined policy that outputs actions conditioned on fused inputs and noise as predicted by the diffusion head

Credit: https://medium.com/@keivalyap/building-vla-models-from-scratch-ii-0180020dbc85