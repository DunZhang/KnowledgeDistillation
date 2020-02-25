import abc


class Evaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self):
        pass
