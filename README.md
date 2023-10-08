# Harnessing Text-to-Image Attention Prior for Reference-based Multi-view Image Synthesis

Codes and datasets will be released soon.

## Abstract

This paper explores the domain of multi-view image synthesis,
aiming to create specific image elements or entire
scenes while ensuring visual consistency with reference images.
We categorize this task into two approaches: local
synthesis, guided by structural cues from reference images
(Reference-based inpainting, Ref-inpainting), and global synthesis,
which generates entirely new images based solely on
reference examples (Novel View Synthesis, NVS). In recent
years, Text-to-Image (T2I) generative models have gained
attention in various domains. However, adapting them for
multi-view synthesis is challenging due to the intricate correlations
between reference and target images. To address
these challenges efficiently, we introduce Attention Reactivated
Contextual Inpainting (ARCI), a unified approach that
reformulates both local and global reference-based multiview
synthesis as contextual inpainting, which is enhanced
with pre-existing attention mechanisms in T2I models. Formally,
self-attention is leveraged to learn feature correlations
across different reference views, while cross-attention
is utilized to control the generation through prompt tuning.
Our contributions of ARCI, built upon the StableDiffusion
fine-tuned for text-guided inpainting, include skillfully handling
difficult multi-view synthesis tasks with off-the-shelf
T2I models, introducing task and view-specific prompt tuning
for generative control, achieving end-to-end Ref-inpainting,
and implementing block causal masking for autoregressive
NVS. We also show the versatility of ARCI by extending it
to multi-view generation for superior consistency with the
same architecture, which has also been validated through
extensive experiments. Codes and models will be released in
https://github.com/ewrfcas/ARCI.