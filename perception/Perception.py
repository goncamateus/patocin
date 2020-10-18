from abc import ABC, abstractmethod

class Perception(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def Perceive(self, data):
        pass
