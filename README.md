# KeyMorph
KeyMorph: Learning Parameter-Efficient Network Activation for Deep Steganography
Official PyTorch implementation of KeyMorph — a key-driven morphing framework that transforms publicly available cover networks into steganographic systems via lightweight key transmission.
📄 Paper: KeyMorph: Learning Parameter-Efficient Network Activation for Deep Steganography
👥 Authors: Fanye Kong, Yu Zheng*, Lei Chen, Jie Zhou, Jiwen Lu
🏛️ Department of Automation, Tsinghua University
🔑 Highlights
Parameter-Efficient Activation: Morph a standard cover network into a steganographic encoder/decoder using keys smaller than 1 KB, eliminating the need to transmit massive model parameters.
Key-Driven Convolution (KD-Conv): Locally partition network weights into non-overlapping segments and enable selective parameter activation through cryptographic keys.
Structural & Functional Indistinguishability: The cover network (e.g., image denoising) is indistinguishable from ordinary CNNs, achieving stronger concealment against both input-output fingerprinting and structure examination attacks.
State-of-the-Art Performance: Superior visual quality, undetectability, and security on DIV2K, COCO, and ImageNet benchmarks.
📦 What's Available Now
✅ Test / Inference Code — Evaluate pre-trained KeyMorph models on standard steganography benchmarks.
🔒 Training Code — Will be released upon paper publication.
