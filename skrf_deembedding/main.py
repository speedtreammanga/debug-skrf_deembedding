from typing import List, Dict
from itertools import combinations
import skrf
import numpy as np
import matplotlib.pyplot as plt


def plot_networks(
    ntwks: List[skrf.Network], labels: List[str], nb_ports: int, delta: bool = False
):
    """
    Plot the magnitude of S-parameters for a list of networks.

    Args:
        ntwks (List[skrf.Network]): A list of scikit-rf Network objects to plot.
        labels (List[str]): A list of labels for each network, corresponding to the order in `ntwks`.
        nb_ports (int): The number of ports for the networks.
        delta (bool, optional): If True, plot the magnitude difference between the first two networks in the list. Default is False.

    Returns:
        None

    Notes:
        - If `delta` is True, only the first two networks in the list will be used for plotting the magnitude difference.
        - The function creates a grid of subplots, with each subplot displaying the magnitude of one S-parameter.
        - The subplot titles are labeled as "S<row><col>", e.g., S11, S21, S32, etc.
        - A legend is included in each subplot to distinguish the different networks.
    """
    fig, axs = plt.subplots(nb_ports, nb_ports, figsize=(15, 10))
    fig.suptitle("Magnitude of S-Parameters")

    # Define a list of S-parameter indices for convenience
    s_params = [(i, j) for i in range(nb_ports) for j in range(nb_ports)]

    # Plot each S-parameter magnitude in its respective subplot
    for ax, (i, j) in zip(axs.flatten(), s_params):
        # plot delta curve...
        if delta:
            if len(ntwks) > 2:
                raise ValueError(
                    f"Cant plot Delta, len(networks) = {len(ntwks)}. Should be 2."
                )
            else:
                sparams_a = ntwks[0].s[:, i, j]
                sparams_b = ntwks[1].s[:, i, j]
                freqs_a = ntwks[0].f
                freqs_b = ntwks[1].f
                if len(freqs_a) != len(freqs_b):
                    raise ValueError("frequency mismatch between the two networks.")
                ax.plot(freqs_a, sparams_a - sparams_b, label=labels[0])
        # plot networks...
        else:
            for ntwk, label in zip(ntwks, labels):
                # only plot if network as i,j in its sparams list (eg, dont plot s33 for a 2-port network)
                if (ntwk.number_of_ports - 1) >= i and (ntwk.number_of_ports - 1) >= j:
                    ax.plot(
                        ntwk.f, ntwk.s[:, i, j], label=label, linewidth=0.85, alpha=0.8
                    )
        ax.set_title(f"S{i+1}{j+1}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend()


def cascade(network: skrf.Network, fixtures: List[skrf.Network]) -> skrf.Network:
    """
    Cascade a multi-port DUT (device under test) network with fixtures connected to each port.

    Args:
        network (skrf.Network): The multi-port DUT network.
        fixtures (List[skrf.Network]): A list of 2-port fixture networks, one for each port of the DUT.
            The order of the fixtures in the list should match the port order of the DUT.

    Returns:
        skrf.Network: The cascaded network with the fixtures de-embedded from the DUT.

    Notes:
        - This function assumes that the DUT and fixtures have the same frequency range and impedance.
        - The number of fixtures must match the number of ports on the DUT.
    """
    if len(fixtures) != network.number_of_ports:
        raise ValueError(
            "The number of fixtures must match the number of ports on the DUT."
        )

    unique_ports_combinations = list(combinations(range(network.number_of_ports), 2))
    cascaded_subnetworks: List[skrf.Network] = []

    for port_a, port_b in unique_ports_combinations:
        subnetwork_a_b = network.subnetwork(ports=[port_a, port_b])
        cascaded_subnetwork_a_b = fixtures[port_a] ** subnetwork_a_b ** fixtures[port_b]
        cascaded_subnetwork_a_b.name = subnetwork_a_b.name
        cascaded_subnetworks.append(cascaded_subnetwork_a_b)

    reconstructed_cascaded_network = skrf.n_twoports_2_nport(
        cascaded_subnetworks, nports=network.number_of_ports
    )
    reconstructed_cascaded_network.name = "cascaded" + network.name

    return reconstructed_cascaded_network


def deembed(network: skrf.Network, fixtures: List[skrf.Network]) -> skrf.Network:
    """
    De-embed a multi-port network from a set of fixtures connected to each port.

    Args:
        network (skrf.Network): The multi-port network to be de-embedded.
        fixtures (List[skrf.Network]): A list of 2-port fixture networks, one for each port of the network.
            The order of the fixtures in the list should match the port order of the network.

    Returns:
        skrf.Network: The de-embedded network with the effects of the fixtures removed.

    Notes:
        - This function assumes that the network and fixtures have the same frequency range and impedance.
        - The number of fixtures must match the number of ports on the network.
    """
    unique_ports_combinations = list(combinations(range(network.number_of_ports), 2))

    deembedded_subnetworks: List[skrf.Network] = []

    for port_a, port_b in unique_ports_combinations:
        subnetwork_a_b = network.subnetwork(ports=[port_a, port_b])
        deembedded_subnetwork_a_b = (
            fixtures[port_a].inv ** subnetwork_a_b ** fixtures[port_b].inv
        )
        deembedded_subnetwork_a_b.name = subnetwork_a_b.name
        deembedded_subnetworks.append(deembedded_subnetwork_a_b)

    reconstructed_deembedded_network = skrf.n_twoports_2_nport(
        deembedded_subnetworks, nports=network.number_of_ports
    )
    reconstructed_deembedded_network.name = "deembedded" + network.name

    return reconstructed_deembedded_network


def main():
    """
    This function demonstrates the usage of the `cascade` and `deembed` functions.

    The main steps performed in this function are:

    1. Load a device under test (DUT) network and a fixture network from scikit-rf's data.
    2. Create a "nudged" fixture network by adding random noise to the original fixture network.
    3. Cascade the DUT network with the original fixture networks using the `cascade` function.
    4. De-embed the original fixture networks from the cascaded DUT network using the `deembed` function.
    5. Plot the magnitude of the S-parameters for the DUT, cascaded DUT, de-embedded DUT, and the fixture network.
    6. Plot the magnitude difference between the original DUT and the de-embedded DUT.

    This function serves as an example to illustrate the cascading and de-embedding processes
    and compare the results with the original DUT network.
    """
    # create dut network
    dut: skrf.Network = skrf.data.tee
    # create fixture network
    fixture: skrf.Network = skrf.data.wr2p2_line
    # create nudged fixture network
    nudged_fixture: skrf.Network = fixture.copy()
    # add noise to nudged fixture network...
    noise_amp = 0.2
    noise = noise_amp * np.random.randn(
        len(nudged_fixture.f), nudged_fixture.s.shape[1], nudged_fixture.s.shape[2]
    )
    nudged_fixture.s = nudged_fixture.s + noise

    # nudged_fixture vs fixture
    # plot_networks(ntwks=[fixture, nudged_fixture], labels=["fixture", "nudged_fixture"], nb_ports=2)
    # plot_networks(
    #     ntwks=[dut, nudged_dut, fixture, nudged_fixture],
    #     labels=["dut", "nudged_dut", "fixture", "nudged_fixture"],
    #     nb_ports=3,
    # )

    fixtures = [fixture, fixture, fixture]
    # nudged_fixtures = [nudged_fixture, nudged_fixture, nudged_fixture]

    cascaded_dut: skrf.Network = cascade(
        dut, fixtures
    )  # build cascaded dut from dut network and fixture networks
    deembedded_dut = deembed(
        cascaded_dut, fixtures
    )  # deembed fixture networks from cascaded dut network

    # plot networks magnitude...
    plot_networks(
        ntwks=[dut, cascaded_dut, deembedded_dut, fixture],
        labels=["dut", "cascaded", "deembedded", "fixture"],
        nb_ports=dut.number_of_ports,
    )
    # plot magnitude delta(dut, deembedded dut)
    plot_networks(
        ntwks=[dut, deembedded_dut],
        labels=["delta"],
        nb_ports=dut.number_of_ports,
        delta=True,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
