 
### Stage 1: The core mechanics

**1. DDPM on 2D toy data**
Train a small MLP to generate points from a 2D shape like a spiral or two moons. Implement the forward noising process, the noise-prediction loss, and the sampling loop yourself.
- *You learn:* the whole algorithm with nothing hidden, and it trains in a minute on a CPU.
- *Do this too:* plot the sampling trajectories and the learned vector field. This makes the denoise-then-add-noise question from earlier very concrete.

**2. DDPM on MNIST, then CIFAR-10**
Build a small U-Net yourself, with timestep embeddings, residual blocks, and attention at low resolution.
- *You learn:* the architecture side, and how much training time and tuning images actually need.
- *Done when:* CIFAR-10 samples are recognisable and you've computed FID.

### Stage 2: Sampling and control

**3. Samplers**
Using your CIFAR model, implement DDIM, then deterministic Euler and Heun ODE samplers. Compare quality versus number of steps.
- *You learn:* that training and sampling are separate choices, and why deterministic samplers need fewer steps.
- *Stretch:* rewrite everything in the EDM (Karras et al.) formulation, which unifies most of the field.

**4. Class conditioning and classifier-free guidance**
Add class labels, randomly drop them during training, and implement CFG at sampling time. Sweep the guidance scale and plot FID against it.
- *You learn:* the fidelity–diversity trade-off from the MIT slide, measured yourself.

**5. Flow matching / rectified flow**
Reimplement projects 1 and 2 with flow matching instead of DDPM.
- *You learn:* the formulation most new models use (e.g. Stable Diffusion 3, Flux), and how much simpler it is.

### Stage 3: Scaling up

**6. Latent diffusion**
Take a pretrained VAE (e.g. the Stable Diffusion one), encode a dataset like CelebA into latents, and train your diffusion model in latent space.
- *You learn:* why modern models don't diffuse in pixel space.

**7. Diffusion Transformer (DiT)**
Replace the U-Net from project 6 with a transformer over latent patches, using adaLN conditioning.
- *You learn:* the architecture behind current large models.

**8. Taking apart Stable Diffusion**
Load a pretrained model's components with `diffusers`, but write your own sampling loop, CFG, img2img and inpainting. Then fine-tune it with LoRA or DreamBooth on your own images.
- *You learn:* how a real text-to-image system fits together.

### Stage 4: Distillation and beyond

**9. Distillation**
Pick one, or do both:
- **Fast sampling:** progressive distillation or a consistency model, on your CIFAR model from Stage 2.
- **Score distillation:** use a pretrained model to optimise a single 2D image with SDS, then with VSD, and compare. This is a cheap way to see the oversaturation problem and VSD's fix before trying 3D.

### Practical notes

- Projects 1–5 run on a single consumer GPU. For 6–9, the GPU nodes at GWDG would make a big difference.
- Good references to check your code against: Lucidrains' `denoising-diffusion-pytorch`, the official EDM repo, and the original DDPM and DiT papers.
- Keep one codebase that grows across projects rather than starting fresh each time. By the end you'll have your own small diffusion library, which is a good thing to show in job applications.
 