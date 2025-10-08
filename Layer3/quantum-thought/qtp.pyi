from typing import List, Tuple
import numpy as np

__version__: str

class ThoughtZone:
    name: str
    type: str
    color: str

    @property
    def position(self) -> Tuple[float, float, float]: ...

    @property
    def position_np(self) -> np.ndarray: ...

    @property
    def scale(self) -> Tuple[float, float, float]: ...

    @property
    def scale_np(self) -> np.ndarray: ...

    def __repr__(self) -> str: ...


class Agent:
    id: str
    maxSteps: int
    stepSize: float
    trailStartColor: str
    trailEndColor: str

    @property
    def position(self) -> Tuple[float, float, float]: ...

    @property
    def position_np(self) -> np.ndarray: ...

    def __repr__(self) -> str: ...


class QuantumThoughtPipeline:
    def __init__(self) -> None: ...
    def build_field(self) -> None: ...
    def get_zones(self) -> List[ThoughtZone]: ...
    def get_agent(self) -> Agent: ...
