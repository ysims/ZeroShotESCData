import torch.nn as nn

# Ideal for networks that use sigmoid or tanh
def weights_init_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            m.bias.data.fill_(0)


# Ideal for networks that use sigmoid or tanh
def weights_init_xavier_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.5)
        if m.bias is not None:
            m.bias.data.fill_(0)


# Ideal for networks that use ReLU
def weights_init_kaiming_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0)


# Ideal for networks that use ReLU
def weights_init_kaiming_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0)
