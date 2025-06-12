import torch
from mnist_encoding import genome_to_net

def fitness(genome):
    try:
        net = genome_to_net(genome).to(DEVICE)
        x, y = next(iter(train_loader))
        logits = net(x.view(x.size(0), -1).to(DEVICE))   # dummy pass
    except Exception:
        return 0.0        # malformed phenotype → dead genome

    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    correct = total = 0
    for _ in range(EPOCHS):
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)

            logits = net(x)
            loss   = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total  # accuracy 0-1