import abc


class IEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
