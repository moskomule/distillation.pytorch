from homura import trainers
from homura.callbacks import metric_callback_decorator
from homura.utils import Map
from homura.vision.models.classification import resnet20, wrn28_10, wrn28_2, resnet56
from torch.nn import functional as F

MODELS = {"resnet20": resnet20,
          "wrn28_10": wrn28_10,
          "wrn28_2": wrn28_2,
          "resnet56": resnet56}


@metric_callback_decorator
def kl_loss(data):
    return data['kl_loss']


class DistillationTrainer(trainers.SupervisedTrainer):
    def __init__(self, model, optimizer, loss_f, callbacks, scheduler, teacher_model, temperature, lambda_factor=1):
        super(DistillationTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler)
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.to(self.device)
        self.temperature = temperature
        self.lambda_factor = lambda_factor

    def iteration(self, data):
        input, target = data
        output = self.model(input)

        if self.is_train:
            self.optimizer.zero_grad()
            lesson = self.teacher(input)
            kl_loss = F.kl_div((output / self.temperature).log_softmax(), (lesson / self.temperature).softmax(),
                               reduction="batchmean")
            loss = self.loss_f(output, target) + self.lambda_factor * (self.temperature ** 2) * kl_loss
            loss.backward()
            self.optimizer.step()
        else:
            loss = self.loss_f(output, target)
            kl_loss = 0
        results = Map(loss=loss, output=output, kl_loss=kl_loss)
        return results
