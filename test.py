import torch
import pytorch_metric_learning.losses as losses


metric_loss = losses.ArcFaceLoss(num_classes=5, embedding_size=64)
print(metric_loss)


x = torch.rand(8, 64).float()
y = torch.randint(0, 5, (8,)).long()

print(metric_loss(x, y))
