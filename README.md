# Full-Glow
This repo contains the implementation of *Full-Glow: Fully conditional Glow for more realistic image generation*:
[https://arxiv.org/abs/2012.05846](https://arxiv.org/abs/2012.05846).

Full-Glow extends on previous Glow-based models for conditional image generation by applying conditioning to all Glow operations 
using appropriate conditioning networks. It was applied to the [Cityscapes](https://www.cityscapes-dataset.com/) dataset (label &#8594; photo) for synthesizing street-scene images.


## Quantitative results
Full-Glow was evaluated quantitatively against previous Glow-based models ([C-Glow](https://arxiv.org/abs/1905.13288) and [DUAL-Glow](https://pitt.edu/~sjh95/teaching/related_papers/dual_glow.pdf)) along with the GAN-based model [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) using the [PSPNet](https://github.com/hszhao/semseg)
classifier. With each trained model, we did inference on the Cityscapes validation set 3 times and calculated the PSP scores.

| Model          | Conditional BPD &#8595; | Mean pixel acc. &#8593; | Mean class acc. &#8593; | Mean class IoU  &#8593; |
| -------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| C-Glow v.1     | 2.568                   | 35.02 ± 0.56            | 12.15 ± 0.05            | 7.33 ± 0.09             |
| C-Glow v.2     | 2.363                   | 52.33 ± 0.46            | 17.37 ± 0.21            | 12.31 ± 0.24            |
| Dual-Glow      | 2.585                   | 71.44 ± 0.03            | 23.91 ± 0.19            | 18.96 ± 0.17            |
| pix2pix        | ---                     | 60.56 ± 0.11            | 22.64 ± 0.21            | 16.42 ± 0.06            |
| **Full-Glow**  | **2.345**               | **73.50 ± 0.13**        | **29.13 ± 0.39**        | **23.86 ± 0.30**        |
| *Ground-truth* | *---*                   | *95.97*                 | *84.31*                 | *77.30*                 |



## Visual samples in 512x1024 resolution
<pre>
                      Condition                                               Synthesized
</pre>

<p align="middle">
  <img src="figs/high_res/diverse/jena_000066_000019/segment.png" width="473" />
  <img src="figs/high_res/diverse/jena_000066_000019/sample_3.png" width="473" />

  <img src="figs/high_res/diverse/aachen_000058_000019/segment.png" width="473" />
  <img src="figs/high_res/diverse/aachen_000058_000019/sample_5.png" width="473" />
  
  <img src="figs/high_res/single/aachen_000011_000019/segment.png" width="473" />
  <img src="figs/high_res/single/aachen_000011_000019/sample_3.png" width="473" />
  
  <img src="figs/high_res/single/aachen_000096_000019/segment.png" width="473" />
  <img src="figs/high_res/single/aachen_000096_000019/sample_6.png" width="473" />

  <img src="figs/high_res/single/aachen_000156_000019/segment.png" width="473" />
  <img src="figs/high_res/single/aachen_000156_000019/sample_6.png" width="473" />

  <img src="figs/high_res/single/aachen_000160_000019/segment.png" width="473" />
  <img src="figs/high_res/single/aachen_000160_000019/sample_9.png" width="473" />
  
  <img src="figs/high_res/single/jena_000078_000019/segment.png" width="473" />
  <img src="figs/high_res/single/jena_000078_000019/sample_3.png" width="473" />
</p>


## Visual examples of content transfer
<pre>
     Desired content            Desired structure           Content applied to           Ground-truth for
                                                               structure                    structure
</pre>

<p align="middle">
  <img src="figs/content_transfer/example_1/content.png" width="230" />
  <img src="figs/content_transfer/example_1/structure.png" width="230" />
  <img src="figs/content_transfer/example_1/result.png" width="230" />
  <img src="figs/content_transfer/example_1/ground_truth.png" width="230" />

  <img src="figs/content_transfer/example_2/content.png" width="230" />
  <img src="figs/content_transfer/example_2/structure.png" width="230" />
  <img src="figs/content_transfer/example_2/result.png" width="230" />
  <img src="figs/content_transfer/example_2/ground_truth.png" width="230" />
  
  <img src="figs/content_transfer/example_3/content.png" width="230" />
  <img src="figs/content_transfer/example_3/structure.png" width="230" />
  <img src="figs/content_transfer/example_3/result.png" width="230" />
  <img src="figs/content_transfer/example_3/ground_truth.png" width="230" />
</p>
