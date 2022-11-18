import torch
import torch.nn as nn

t = torch.rand(4, 4, 3)
print(t)

num_keep = int(t.shape[1] * (1 - 0.5))

row_perm = torch.rand(t.shape[:2], device=t.device).argsort(1)
print( row_perm)
undo_row_perm = torch.argsort(row_perm, dim=1) # get back indices
row_perm = row_perm[:, :num_keep]
row_perm.unsqueeze_(-1)
row_perm = row_perm.repeat(1, 1, t.shape[2])  # reformat this for the gather operation

t_masked = t.gather(1, row_perm)

# generate the binary mask: 0 is keep, 1 is remove
mask = torch.ones(t.shape[:2], device=t.device)
mask[:, :num_keep] = 0
mask = torch.gather(mask, dim=1, index=undo_row_perm)


print(t_masked)

print(undo_row_perm)
print(mask)


