# Harnessing Text-to-Image Attention Prior for Reference-based Multi-view Image Synthesis

Codes and datasets will be released soon.

[Project Page](https://ewrfcas.github.io/ARCI/)

## Abstract

This paper explores the domain of multi-view image synthesis.
We categorize this task into two approaches: local
synthesis, guided by structural cues from reference images
(Reference-based inpainting, Ref-inpainting), and global synthesis,
which generates entirely new images based solely on
reference examples (Novel View Synthesis, NVS). 
We introduce Attention Reactivated
Contextual Inpainting (ARCI), a unified approach that
reformulates both local and global reference-based multiview
synthesis as contextual inpainting, which is enhanced
with pre-existing attention mechanisms in Text-to-Image (T2I) models. 
Formally,
self-attention is leveraged to learn feature correlations
across different reference views, while cross-attention
is utilized to control the generation through prompt tuning.
We also show the versatility of ARCI by extending it
to multi-view generation for superior consistency with the
same architecture. 
Codes and models will be released in
https://github.com/ewrfcas/ARCI.