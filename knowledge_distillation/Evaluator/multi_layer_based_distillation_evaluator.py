from .evaluator import Evaluator
import os
import logging

logger = logging.getLogger(__name__)


class MultiLayerBasedDistillationEvaluator(Evaluator):
    def __init__(self, save_dir, save_step=None, print_loss_step=20):
        super(MultiLayerBasedDistillationEvaluator, self).__init__()
        self.save_step = save_step
        self.print_loss_step = print_loss_step
        self.save_dir = os.path.abspath(save_dir)
        self.loss_fw = None

        os.makedirs(self.save_dir, exist_ok=True)
        if save_dir and save_step:
            self.loss_fw = open(os.path.join(save_dir, "LossValue.txt"), "w", encoding="utf8")

    def evaluate(self, teacher_model, student_model, dev_data, dev_data_adaptor, epoch, step, loss_value):
        if step > 0 and step % self.print_loss_step == 0:
            logger.info("epoch:{},\tstep:{},\tloss value:{}".format(epoch, step, loss_value))
            if self.save_dir and self.save_step:
                self.loss_fw.write("epoch:{},\tstep:{},\tloss value:{}\n".format(epoch, step, loss_value))
                self.loss_fw.flush()
        if self.save_dir and self.save_step:
            if step > 0 and step % self.save_step == 0:
                save_path = os.path.join(self.save_dir, str(epoch) + "-" + str(step))
                os.makedirs(save_path, exist_ok=True)
                student_model.save_pretrained(save_path)


if __name__ == "__main__":
    logger.info("epoch:{}\tstep:{}\tloss value:{}".format(1, 2, 3))
