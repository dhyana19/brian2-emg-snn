from brian2 import *
from windowing import DEFAULT_SNN_ENCODER_MODE, get_snn_input_channels


INPUT_THRESHOLD = 0.05
HIDDEN_THRESHOLD = 0.15
OUTPUT_THRESHOLD = 0.15

INPUT_WEIGHT_BASE = 2.0
INPUT_WEIGHT_SPAN = 0.5
OUTPUT_WEIGHT_BASE = 0.4
OUTPUT_WEIGHT_SPAN = 0.4

HIDDEN_INHIB_STRENGTH = 0.2
HIDDEN_INHIB_PROB = 0.05
OUTPUT_INHIB_STRENGTH = 0.3


def build_snn(
    n_inputs=None,
    n_hidden=80,
    n_outputs=2,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
    record_spike_trains=False
):

    if n_inputs is None:
        n_inputs = get_snn_input_channels(encoder_mode=encoder_mode)

    start_scope()

    # -----------------------------
    # Input neurons (driven by EMG)
    # -----------------------------
    eqs_in = '''
    dv/dt = (-v + input_signal(t,i)) / (5*ms) : 1
    '''

    input_group = NeuronGroup(
        n_inputs,
        eqs_in,
        threshold=f'v > {INPUT_THRESHOLD}',
        reset='v = 0',
        method='euler'
    )

    # -----------------------------
    # Hidden neuron model
    # -----------------------------
    eqs = '''
    dv/dt = -v / (10*ms) : 1
    '''

    hidden = NeuronGroup(
        n_hidden,
        eqs,
        threshold=f'v > {HIDDEN_THRESHOLD}',
        reset='v = 0',
        method='euler'
    )

    # -----------------------------
    # Output neurons
    # -----------------------------
    output = NeuronGroup(
        n_outputs,
        eqs,
        threshold=f'v > {OUTPUT_THRESHOLD}',
        reset='v = 0',
        method='euler'
    )

    # -----------------------------
    # Input → Hidden synapses
    # -----------------------------
    syn_in = Synapses(input_group, hidden, 'w : 1', on_pre='v_post += w')
    syn_in.connect()

    # Slightly stronger input drive helps the hidden layer stay informative.
    syn_in.w = f'{INPUT_WEIGHT_BASE} + rand()*{INPUT_WEIGHT_SPAN}'


    # -----------------------------
    # Hidden → Output synapses
    # -----------------------------
    syn_out = Synapses(hidden, output, 'w : 1', on_pre='v_post += w')
    syn_out.connect()

    syn_out.w = f'{OUTPUT_WEIGHT_BASE} + rand()*{OUTPUT_WEIGHT_SPAN}'


    # -----------------------------
    # Hidden layer lateral inhibition
    # -----------------------------
    inhib_hidden = Synapses(
        hidden,
        hidden,
        on_pre=f'v_post -= {HIDDEN_INHIB_STRENGTH}'
    )
    inhib_hidden.connect(condition='i != j', p=HIDDEN_INHIB_PROB)


    # -----------------------------
    # Output layer winner-take-all
    # -----------------------------
    inhib_out = Synapses(output, output, on_pre=f'v_post -= {OUTPUT_INHIB_STRENGTH}')
    inhib_out.connect(condition='i != j')


    # -----------------------------
    # Spike monitors (counts only)
    # prevents memory crash
    # -----------------------------
    spike_hidden = SpikeMonitor(hidden, record=record_spike_trains)
    spike_out = SpikeMonitor(output, record=record_spike_trains)


    # -----------------------------
    # Build network
    # -----------------------------
    net = Network(
        input_group,
        hidden,
        output,
        syn_in,
        syn_out,
        inhib_hidden,
        inhib_out,
        spike_hidden,
        spike_out
    )

    return net, input_group, hidden, output, syn_out, spike_hidden, spike_out
