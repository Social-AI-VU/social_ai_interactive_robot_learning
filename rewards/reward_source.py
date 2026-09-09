# rewards/reward_source.py

from abc import ABC, abstractmethod


class RewardSource(ABC):

    @abstractmethod
    def start(self):
        pass