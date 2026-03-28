from abc import ABC, abstractmethod

class Motor(ABC):
    @property
    def position(self):
        """
        Read-only property for the current position of the motor
        """
        raise NotImplementedError("Subclasses must implement this property")

    def move(self, *, speed: float):
        raise NotImplementedError("Subclasses must implement this method")

    def go(self):
        """
        Execute the movement
        """
        raise NotImplementedError("Subclasses must implement this method")
