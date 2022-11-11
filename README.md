## Exploring: Masked Autoencoders Are Scalable Vision Learners

### Contributors

@altansnl, @AGarciaCast, @Frankkie

### TO-DO

- For our experiments, we will use Tiny ImageNet instead of ImageNet-1K. For our baseline model,
we will use ViT-B instead of ViT-Large.
- For MAE ablation experiments, we will only use fine-tuning and not use linear probing. We will
experiment with decoder-depth, decoder-width, and reconstruction-target; to lesser exhaus-
tive extent with respect to the paper (due to time constrains we will not experiment with encoder
with mask-tokens).
- For comparisons with previous results on Tiny ImageNet, we will point to the respective papers and
will not verify their results ourselves.
- We will leave partial fine-tuning out-of-scope.
- We might experiment with transfer learning for downstream tasks of object detection or a classification
task (the amount of experiments will depend on the time and resources constrains).
- The author mention that the performance could be improved if non-vanilla ViT models are used. So
we could try to compare our results with other variations (not only on depth s.t. ViT-L/H).
