import numpy as np
from wegp_bayes.utils.variables import NumericalVariable,CategoricalVariable
from wegp_bayes.utils.input_space import InputSpace

def borehole():
    
    V0 = np.array(np.linspace(0.05,0.15,4))
    V1 = np.array(np.linspace(700,820,4))
    

    config = InputSpace()
    r = NumericalVariable(name='r',lower=100,upper=50000)
    Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
    Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
    Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
    L = NumericalVariable(name='L',lower=1120,upper=1680)
    K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)
    config.add_inputs([r,Tu,Hu,Tl,L,K_w])

    config.add_input(
        CategoricalVariable(name='r_w',levels=np.linspace(0.05,0.15,4))
    )
    config.add_input(
        CategoricalVariable(name='H_l',levels=np.linspace(700,820,4))
    )
    return config






def piston():
    config = InputSpace()
    M = NumericalVariable(name='M',lower=30,upper=60)
    S = NumericalVariable(name='S',lower=0.005,upper=0.02)
    V0 = NumericalVariable(name='V_0',lower=0.002,upper=0.01)
    Ta = NumericalVariable(name='T_a',lower=290,upper=296)
    T0 = NumericalVariable(name='T_0',lower=340,upper=360)
    config.add_inputs([M,S,V0,Ta,T0])
    config.add_input(
        CategoricalVariable(name='k',levels=np.linspace(1000,5000,4))
    )
    config.add_input(
        CategoricalVariable(name='P_0',levels=np.linspace(90000,110000,4))
    )
    return config

