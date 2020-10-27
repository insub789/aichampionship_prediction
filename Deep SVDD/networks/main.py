import sys
sys.path.insert(1, '/workspace/Deep SVDD/')
from networks.baeminNet import baeminNet, baeminNet_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('baeminNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'baeminNet':
        net = baeminNet()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('baeminNet')
    assert net_name in implemented_networks

    ae_net = None
    
    if net_name == 'baeminNet':
        ae_net = baeminNet_Autoencoder()

    return ae_net


