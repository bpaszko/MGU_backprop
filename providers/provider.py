from abc import ABC, abstractmethod


class Provider:
    @abstractmethod
    def __iter__(self):
        pass
        
    @abstractmethod
    def __next__(self):
        pass
