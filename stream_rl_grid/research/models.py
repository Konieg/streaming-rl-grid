"""Interchangeable stochastic models for clean Dyna diagnostics."""

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .environment import ContinuingGridMDP


ModelKey = Tuple[int, int]
Outcome = Tuple[float, int]


class BaseModel:
    name = "base"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.keys: List[ModelKey] = []
        self._key_set = set()

    def _register(self, state: int, action: int) -> ModelKey:
        key = (int(state), int(action))
        if key not in self._key_set:
            self._key_set.add(key)
            self.keys.append(key)
        return key

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        raise NotImplementedError

    def sample_key(self) -> ModelKey:
        if not self.keys:
            raise RuntimeError("cannot sample an empty model")
        return self.keys[int(self.rng.integers(len(self.keys)))]

    def sample(self, key: ModelKey) -> Outcome:
        raise NotImplementedError

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.keys)


class LatestTransitionModel(BaseModel):
    name = "latest"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.data: Dict[ModelKey, Outcome] = {}

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        key = self._register(state, action)
        self.data[key] = (float(reward), int(next_state))

    def sample(self, key: ModelKey) -> Outcome:
        return self.data[key]

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        return {self.data[key]: 1.0}


class EmpiricalModel(BaseModel):
    name = "empirical"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.counts: Dict[ModelKey, Counter] = defaultdict(Counter)

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        key = self._register(state, action)
        self.counts[key][(float(reward), int(next_state))] += 1

    def sample(self, key: ModelKey) -> Outcome:
        outcomes = list(self.counts[key])
        counts = np.asarray([self.counts[key][outcome] for outcome in outcomes], dtype=np.float64)
        return outcomes[int(self.rng.choice(len(outcomes), p=counts / counts.sum()))]

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        total = float(sum(self.counts[key].values()))
        return {outcome: count / total for outcome, count in self.counts[key].items()}


class ExponentialRecencyModel(BaseModel):
    name = "ema"

    def __init__(self, decay: float = 0.98, seed: int = 0):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        super().__init__(seed)
        self.decay = float(decay)
        self.weights: Dict[ModelKey, Dict[Outcome, float]] = defaultdict(dict)

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        key = self._register(state, action)
        table = self.weights[key]
        for outcome in tuple(table):
            table[outcome] *= self.decay
            if table[outcome] < 1e-12:
                del table[outcome]
        outcome = (float(reward), int(next_state))
        table[outcome] = table.get(outcome, 0.0) + 1.0

    def sample(self, key: ModelKey) -> Outcome:
        outcomes = list(self.weights[key])
        values = np.asarray([self.weights[key][outcome] for outcome in outcomes])
        return outcomes[int(self.rng.choice(len(outcomes), p=values / values.sum()))]

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        total = float(sum(self.weights[key].values()))
        return {outcome: weight / total for outcome, weight in self.weights[key].items()}


class SlidingWindowModel(BaseModel):
    name = "sliding"

    def __init__(self, window: int = 100, seed: int = 0):
        if window < 1:
            raise ValueError("window must be positive")
        super().__init__(seed)
        self.window = int(window)
        self.data: Dict[ModelKey, Deque[Outcome]] = defaultdict(lambda: deque(maxlen=self.window))

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        key = self._register(state, action)
        self.data[key].append((float(reward), int(next_state)))

    def sample(self, key: ModelKey) -> Outcome:
        values = self.data[key]
        return values[int(self.rng.integers(len(values)))]

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        counts = Counter(self.data[key])
        total = float(sum(counts.values()))
        return {outcome: count / total for outcome, count in counts.items()}


class OracleModel(BaseModel):
    name = "oracle"

    def __init__(self, mdp: ContinuingGridMDP, seed: int = 0):
        super().__init__(seed)
        self.mdp = mdp
        for state in range(len(mdp.states)):
            for action in range(mdp.num_actions):
                self._register(state, action)

    def observe(self, state: int, action: int, reward: float, next_state: int) -> None:
        del state, action, reward, next_state

    def distribution(self, key: ModelKey) -> Dict[Outcome, float]:
        state, action = key
        result: Dict[Outcome, float] = {}
        for probability, next_state, reward, _, _ in self.mdp.transition_distribution(state, action):
            outcome = (float(reward), int(next_state))
            result[outcome] = result.get(outcome, 0.0) + probability
        return result

    def sample(self, key: ModelKey) -> Outcome:
        distribution = self.distribution(key)
        outcomes = list(distribution)
        probabilities = np.asarray([distribution[outcome] for outcome in outcomes])
        return outcomes[int(self.rng.choice(len(outcomes), p=probabilities))]


def model_total_variation(model: BaseModel, mdp: ContinuingGridMDP) -> float:
    """Mean TV error over state-actions that the learned model has observed."""
    if not model.keys:
        return float("nan")
    errors = []
    oracle = OracleModel(mdp)
    for key in model.keys:
        predicted = model.distribution(key)
        target = oracle.distribution(key)
        support = set(predicted) | set(target)
        errors.append(0.5 * sum(abs(predicted.get(x, 0.0) - target.get(x, 0.0)) for x in support))
    return float(np.mean(errors))
