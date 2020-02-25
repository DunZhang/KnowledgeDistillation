from .evaluator import Evaluator


class TinyBERTEvaluator(Evaluator):
    def __init__(self, save_dir, save_step=2000):
        super(TinyBERTEvaluator, self).__init__()
        self.save_step = save_step
        self.save_dir = save_dir

    def evaluate(self, teacher_model, student_model, evaluate_data, epoch, step):
        if step > 0 and step % self.save_step == 0:
            student_model.save(self.save_dir)
