from .evaluator import Evaluator
import os


class TinyBERTEvaluator(Evaluator):
    def __init__(self, save_dir, save_step=2000):
        super(TinyBERTEvaluator, self).__init__()
        self.save_step = save_step
        self.save_dir = save_dir
        self.loss_fw = None
        if save_dir and save_step:
            self.loss_fw = open(os.path.join(save_dir, "loss.txt"), "w", encoding="utf8")

    def evaluate(self, teacher_model, student_model, evaluate_data, epoch, step, loss_value):
        if step % 10 == 0:
            print(epoch, step, loss_value)
            if self.save_dir and self.save_step:
                self.loss_fw.write(str(epoch) + "-" + str(step) + "-" + str(loss_value) + "\n")
                self.loss_fw.flush()
        if self.save_dir and self.save_step:
            if step > 0 and step % self.save_step == 0:
                save_path = os.path.join(self.save_dir, str(epoch) + "-" + str(step))
                os.makedirs(save_path, exist_ok=True)
                student_model.save(save_path)
