import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

class AbstractEnvRunner2(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = ((nenv*nsteps,) + env.observation_space[0].shape,(nenv*nsteps,) + env.observation_space[1].shape)
        self.obs = [np.zeros((nenv,) + env.observation_space[0].shape, dtype=env.observation_space[0].dtype.name),
                    np.zeros((nenv,) + env.observation_space[1].shape, dtype=env.observation_space[1].dtype.name)]
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError
