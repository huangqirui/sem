
from neuron_dendritic import *


def get_neuron(type=''):
    if type == 'space_multistep_2d':
        return MultiStepDendriticNeuron2DSpace
    elif type == 'space_multistep_repeat_1d':
        return MultiStepRepeatedDendriticNeuron1DSpace

    elif type == 'time_multistep_2d':
        return MultiStepDendriticNeuron2DTime
    elif type == 'time_multistep_repeat_1d':
        return MultiStepRepeatedDendriticNeuron1DTime

    elif type == 'multistep_split_1d':
        return MultiStepSplitDendriticNeuron1D
    elif type == 'repeat_1d':
        return RepeatedDendriticNeuron1D
