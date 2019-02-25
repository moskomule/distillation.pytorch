import torch
from homura import trainers
from homura.utils import Map
from torch.nn import functional as F


def hot_logsoftmax(input: torch.Tensor, temperature: float, dim=-1) -> torch.Tensor:
    """ Logsoftmax with temperature 
    
    :param input: input logits
    :param temperature: temperature parameter
    :param dim:
    :return: 
    """
    return input / temperature - (input / temperature).logsumexp(dim=dim, keepdim=True)


def hot_cross_entropy(input, target, temperature):
    return F.nll_loss(hot_logsoftmax(input, temperature), target)


torch.Tensor.hot_logsoftmax = hot_logsoftmax


class DistillationTrainer(trainers.SupervisedTrainer):
    def __init__(self, model, optimizer, loss_f, callbacks, scheduler, teacher_model, temperature):
        super(DistillationTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler)
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.to(self.device)
        self.temperature = temperature

    def iteration(self, data):
        input, target = data
        output = self.model(input)
        with torch.no_grad():
            lesson = self.teacher(input)
        loss = self.loss_f(output, target) + (self.temperature ** 2) * F.kl_div(output,
                                                                                lesson.hot_logsoftmax(self.temperature))
        results = Map(loss=loss, output=output)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return results
