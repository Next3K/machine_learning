from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> int:
        pass
