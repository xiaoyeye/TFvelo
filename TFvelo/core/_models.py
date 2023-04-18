from typing import Dict, List, Tuple, Union

import numpy as np
from numpy import ndarray

from ._arithmetic import invert
from ._base import DynamicsBase


# TODO: Improve parameter names: alpha -> transcription_rate; beta -> splicing_rate;
# gamma -> degradation_rate
# TODO: Handle cases beta = 0, gamma == 0, beta == gamma
class SplicingDynamics(DynamicsBase):
    """Splicing dynamics.

    Arguments
    ---------
    alpha
        Transcription rate.
    beta
        Translation rate.
    gamma
        Splicing degradation rate.
    initial_state
        Initial state of system. Defaults to `[0, 0]`.

    Attributes
    ----------
    alpha
        Transcription rate.
    beta
        Translation rate.
    gamma
        Splicing degradation rate.
    initial_state
        Initial state of system. Defaults to `[0, 0]`.
    u0
        Initial abundance of unspliced RNA.
    s0
        Initial abundance of spliced RNA.

    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        omega: float,
        theta: float,
        gamma: float,
        array_flag = False
        #initial_state: Union[List, ndarray] = [0, 0],
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.theta = theta
        self.array_flag = array_flag

        if self.array_flag:
            self.alpha=self.alpha.reshape(-1,1)
            self.beta=self.beta.reshape(-1,1)
            self.gamma=self.gamma.reshape(-1,1)
            self.omega=self.omega.reshape(-1,1)
            self.theta=self.theta.reshape(-1,1)

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, val):
        if isinstance(val, list) or (isinstance(val, ndarray) and (val.ndim == 1)):
            self.u0 = val[0]
            self.s0 = val[1]
        else:
            self.u0 = val[:, 0]
            self.s0 = val[:, 1]
        self._initial_state = val

    def get_solution(
        self, t: ndarray, stacked: bool = True, with_keys: bool = False
    ) -> Union[Dict, ndarray]:
        """Calculate solution of dynamics.

        Arguments
        ---------
        t
            Time steps at which to evaluate solution.
        stacked
            Whether to stack states or return them individually. Defaults to `True`.
        with_keys
            Whether to return solution labelled by variables in form of a dictionary.
            Defaults to `False`.

        Returns
        -------
        Union[Dict, ndarray]
            Solution of system. If `with_keys=True`, the solution is returned in form of
            a dictionary with variables as keys. Otherwise, the solution is given as
            a `numpy.ndarray` of form `(n_steps, 2)`.
        """
        if self.array_flag:
            t = t.reshape(1,-1)

        phi = np.arctan(self.omega/self.gamma)
        tmp1 = self.omega * t + self.theta
        tmp2 = np.sqrt(self.omega*self.omega + self.gamma*self.gamma)

        y = self.alpha * np.sin(tmp1) + self.beta
        WX = self.alpha * tmp2 * np.sin(tmp1+phi) + self.beta*self.gamma

        if with_keys:
            return {"WX": WX, "y": y}
        elif not stacked:
            return WX, y
        else:
            if isinstance(t, np.ndarray) and t.ndim == 2:
                return np.stack([WX, y], axis=2)
            else:
                return np.column_stack([WX, y])

