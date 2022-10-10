import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from prob_transformer.utils.summary import SummaryDict


def run_epoch(rank, cfg, data_iter, model, geco_criterion=None, opti=None):
    model_type = cfg.model.model_type

    if opti is None:
        model.eval()
        is_train = False
    else:
        model.train()
        is_train = True

    epoch_summary = SummaryDict()

    batch_size_list = []
    seq_len_list = []

    if cfg.train.amp and cfg.train.grad_scale:
        scaler = amp.GradScaler(init_scale=cfg.train.grad_scale, growth_factor=2.0, backoff_factor=0.5,
                                growth_interval=2000, enabled=True)

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1).to(rank)

    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(data_iter):

            if is_train and i > cfg.train.iter_per_epoch:
                break

            if is_train:
                for param in model.parameters():
                    param.grad = None

            if "encoder" in model_type:
                batch_size_list.append(batch.src_seq.size()[0])
                seq_len_list.append(batch.src_seq.size()[1])
            else:
                batch_size_list.append(batch.trg_seq.size()[0])
                seq_len_list.append(batch.trg_seq.size()[1])

            with amp.autocast(enabled=cfg.train.amp):

                if "encoder" in model_type:
                    if "prob" in model_type and is_train:
                        pred_seq, z_lists = model(batch.src_seq, batch.src_len,
                                                  post_trg_seq=batch.post_seq,
                                                  infer_mean=False)
                    else:
                        pred_seq = model(batch.src_seq, batch.src_len, infer_mean=True)

                elif "decoder" in model_type:  # decoder only
                    props, scaffold = None, None
                    if cfg.data.type == "mol":
                        if cfg.data.mol.props:
                            props = batch.props

                    if "prob" in model_type and is_train:
                        pred_seq, z_lists = model(post_trg_seq=batch.post_seq,
                                                  trg_shf_seq=batch.trg_shf_seq,
                                                  trg_len=batch.trg_len,
                                                  props=props, infer_mean=False)
                    else:
                        pred_seq = model(trg_shf_seq=batch.trg_shf_seq, trg_len=batch.trg_len,
                                         props=props, infer_mean=True)

            sequence_loss = criterion(pred_seq.contiguous().view(-1, pred_seq.size(-1)),
                                      batch.trg_seq.contiguous().view(-1)).view(batch.trg_seq.shape)
            sequence_loss = torch.sum(sequence_loss, dim=-1) / batch.trg_len
            sequence_loss = sequence_loss.mean()

            if is_train and "prob" in model_type:
                epoch_summary["pre_geco_loss"] = sequence_loss.detach()

                crit_set = z_lists, sequence_loss
                trg_set = batch.trg_seq, batch.trg_len
                geco_loss, summary = geco_criterion(crit_set, trg_set)

                epoch_summary["geco_loss"] = geco_loss.detach()
                loss = geco_loss
            else:
                loss = sequence_loss

            if opti is not None and 0 < loss.cpu().item():

                if cfg.train.amp and cfg.train.grad_scale:
                    loss = scaler.scale(loss)
                    loss.backward()
                    scaler.unscale_(opti.optimizer)
                else:
                    loss.backward()

                if cfg.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)

                if cfg.train.amp and cfg.train.grad_scale:
                    scaler.step(opti.optimizer)
                    scaler.update()
                else:
                    opti.optimizer.step()

                opti.train_step()
                opti.optimizer.zero_grad()

            epoch_summary["loss"] = loss.detach()
            epoch_summary["step"] = i

            if "prob" in model_type and is_train:
                epoch_summary(summary)

    loss = np.mean(epoch_summary['loss'])
    step = np.max(epoch_summary['step'])
    batch = np.mean(batch_size_list)
    seq_len = np.mean(seq_len_list)
    stats = {"mean_batch_size": batch, "step": step, "mean_seq_len": seq_len, "loss": loss}

    if "prob" in model_type and is_train:
        stats['ma_ce_loss'] = np.mean(epoch_summary['ma_ce_loss'])
        stats['rec_constraint'] = np.mean(epoch_summary['rec_constraint'])
        stats['lamb'] = np.mean(epoch_summary['lamb'])

    return stats, epoch_summary
