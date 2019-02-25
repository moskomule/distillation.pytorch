import torch.nn.functional as F
from homura import optim, lr_scheduler, callbacks, trainers, reporters
from homura.vision.data.loaders import cifar10_loaders
from homura.vision.models.classification import resnet20, wrn28_10
from tqdm import trange
from utils import DistillationTrainer, kl_loss


def main():
    model = {"resnet20": resnet20,
             "wrn28_10": wrn28_10}[args.model](num_classes=10)
    weight_decay = {"resnet20": 1e-4,
                    "wrn28_10": 5e-4}[args.model]
    lr_decay = {"resnet20": 0.1,
                "wrn28_10": 0.2}[args.model]
    train_loader, test_loader = cifar10_loaders(args.batch_size)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR([50, 80], gamma=lr_decay)

    trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, scheduler=scheduler)
    trainer.logger.info("Train the teacher model!")
    for _ in trange(100, ncols=80):
        trainer.train(train_loader)
        trainer.test(test_loader)

    teacher_model = model.eval()
    model = {"resnet20": resnet20,
             "wrn28_10": wrn28_10}[args.model](num_classes=10)
    c = [callbacks.AccuracyCallback(), callbacks.LossCallback(), kl_loss]
    with reporters.TQDMReporter(range(200), callbacks=c) as tq, reporters.TensorboardReporter(c) as tb:
        trainer = DistillationTrainer(model, optimizer, F.cross_entropy, callbacks=[tq, tb],
                                      scheduler=scheduler, teacher_model=teacher_model, temperature=args.temperature)
        trainer.logger.info("Train the student model!")
        for _ in tq:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs

    p = miniargs.ArgumentParser()
    p.add_int("--batch_size", default=128)
    p.add_str("--model", choices=["resnet20", "wrn28_10"])
    p.add_float("--temperature", default=0.1)

    args = p.parse()
    main()
