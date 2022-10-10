import torch


class OptiMaster():
    def __init__(self, model, epochs, iter_per_epoch, optimizer, scheduler, warmup_epochs, lr_low, lr_high, beta1,
                 beta2, weight_decay):

        self.lr_low = lr_low
        self.lr_high = lr_high
        self.scheduler = scheduler
        self.weight_decay = weight_decay

        self.iter_per_epoch = iter_per_epoch
        self.epochs = epochs
        self.model = model

        self.warmup_low = 1e-9
        self.epoch = -1

        init_lr = self.lr_high

        self.optimizer = self._get_optimizer(optimizer, model, lr=init_lr, beta1=beta1, beta2=beta2,
                                             weight_decay=weight_decay)

        if warmup_epochs > 0:
            self.warmup_steps = iter_per_epoch * (warmup_epochs)
            lr_func = lambda step: step / self.warmup_steps
            warmup_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)
            self.warmup_schedule = warmup_schedule
            self.warmup_epochs = warmup_epochs  # + 1
        else:
            self.warmup_epochs = 0
        self.train_epochs = epochs - self.warmup_epochs

        max_train_epoch = self.train_epochs

        main_schedule = self._get_schedule(scheduler, max_epoch=max_train_epoch)
        self.main_schedule = main_schedule

        if warmup_epochs > 0:
            self.optimizer.param_groups[0]['lr'] = self.warmup_low

    def epoch_step(self, epoch):
        self.epoch = epoch  # + 1

        if self.epoch < self.warmup_epochs - 1:
            pass
        elif self.epoch > self.warmup_epochs - 1:
            self.main_schedule.step()

    def train_step(self):
        if self.epoch < self.warmup_epochs - 1:
            self.warmup_schedule.step()

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def config_weight_decay(self, model):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith('scale') or pn.endswith('key_dim_scaler'):
                    no_decay.add(fpn)
                elif 'lagmul' in pn:
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def _get_optimizer(self, optim_name, model, lr, beta1, beta2, weight_decay):

        if self.weight_decay == 0 or self.weight_decay == False:
            params = model.parameters()
        else:
            params = self.config_weight_decay(model)

        if optim_name == "adam":
            return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=1e-9, weight_decay=weight_decay)
        elif optim_name == "adamW":
            return torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=1e-9, weight_decay=weight_decay)
        elif optim_name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, alpha=0.98, momentum=0.1, eps=1e-9, weight_decay=weight_decay)

    def _get_schedule(self, schedule_name, max_epoch):
        if schedule_name == "step":
            train_gamma = (self.lr_low / self.lr_high) ** (1 / max_epoch)
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=train_gamma)

        elif schedule_name == "linear":
            lr_func = lambda epoch: (self.lr_low / self.lr_high - 1) * epoch / max_epoch + 1
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)

        elif schedule_name == "inv_sqrt":
            lr_func = lambda epoch: self.warmup_steps ** 0.5 / (
                    (self.warmup_epochs + epoch) * self.iter_per_epoch) ** 0.5
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)

        elif schedule_name == "const":
            lr_func = lambda epoch: 1
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)

        elif schedule_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, max_epoch, eta_min=self.lr_low)
