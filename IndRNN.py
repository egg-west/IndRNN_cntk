import cntk as C
from cntk.variables import Parameter
from cntk.ops import times, slice, sigmoid, tanh, softplus, relu
#from .typing import Signature
from cntk.internal import _as_tuple
from cntk.initializer import glorot_uniform
from cntk.default_options import get_default_override, default_override_or
from cntk.ops.functions import BlockFunction 
from cntk.layers import RNNStep
from cntk.layers.blocks import _INFERRED, Stabilizer

def IndRNNStep(shape, cell_shape=None, activation=default_override_or(relu),
            init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
            enable_self_stabilization=default_override_or(False),
            name=''):

    activation                = get_default_override(RNNStep, activation=activation)
    init                      = get_default_override(RNNStep, init=init)
    init_bias                 = get_default_override(RNNStep, init_bias=init_bias)
    enable_self_stabilization = get_default_override(RNNStep, enable_self_stabilization=enable_self_stabilization)

    return IndRNNBlock('RNNStep', shape, cell_shape, activation=activation, use_peepholes=False,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)

def IndRNNBlock(type, shape, cell_shape, activation, use_peepholes,
                    init, init_bias,
                    enable_self_stabilization,
                    name=''):
    '''
    Helper to create a recurrent block of type 'LSTM', 'GRU', or RNNStep.
    '''

    has_projection = cell_shape is not None

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape
    if len(shape) != 1 or len(cell_shape) != 1:
        raise ValueError("%s: shape and cell_shape must be vectors (rank-1 tensors)" % type)
        # otherwise we'd need to fix slicing and Param initializers

    stack_axis = -1  # for efficient computation, we stack multiple variables (along the fastest-changing one, to match BS)
    # determine stacking dimensions
    cell_shape_list = list(cell_shape)
    stacked_dim = cell_shape_list[stack_axis]
    cell_shape_list[stack_axis] = stacked_dim * 1
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times
    cell_shape_list[stack_axis] = stacked_dim * 1
    cell_shape_stacked_H = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    b  = Parameter(            cell_shape_stacked,   init=init_bias, name='b')                              # bias
    W  = Parameter(_INFERRED + cell_shape_stacked,   init=init,      name='W')                              # input
    H  = Parameter(            cell_shape_stacked_H, init=init,      name='H')                              # hidden-to-hidden

    Wmr = Parameter(cell_shape + shape, init=init, name='P') if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dh_stabilizer')
    Sdc = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dc_stabilizer')
    Sct = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='c_stabilizer')
    Sht = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='P_stabilizer')

   
    def rnn_step(dh, x):
        dhs = Sdh(dh)  # previous value, stabilized
        ht = activation (times(x, W) + dhs * H + b)
        h = times(Sht(ht), Wmr) if has_projection else \
            ht
        return h

    function = {
        'RNNStep': rnn_step,
    }[type]

    # return the corresponding lambda as a CNTK Function
    return BlockFunction(type, name)(function)

