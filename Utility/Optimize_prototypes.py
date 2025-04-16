import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn


def prototype_loss(prototypes):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()

def create_hypersphere(num_classes, output_dimension, max_epoch=2000):

    prototypes = torch.randn(num_classes, output_dimension)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad = True)
    optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    for epoch in range(max_epoch):
        loss, sep = prototype_loss(prototypes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad = True)
        optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    return prototypes.detach()


if __name__ == '__main__':
    class_matched_points = create_hypersphere(3, 10, ['1', '2', '3'])
    print(class_matched_points)