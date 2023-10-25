def weights_initialization(self):
    '''
    When we define all the modules such as the layers in '__init__()'
    method above, these are all stored in 'self.modules()'.
    We go through each module one by one. This is the entire network,
    basically.
    '''
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


ema = EMA(0.999)
for name, param in model.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)

    # in batch training loop
    # for batch in batches:
    optimizer.step()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = ema(name, param.data)