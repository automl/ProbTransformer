from typing import List
import pathlib
import socket
import numpy as np
import torch

from prob_transformer.utils.supporter import Supporter
from prob_transformer.utils.summary import SummaryDict
from prob_transformer.utils.config_init import cinit
from prob_transformer.utils.torch_utils import count_parameters

from prob_transformer.module.optim_builder import OptiMaster
from prob_transformer.model.probtransformer import ProbTransformer
from prob_transformer.data.iterator import MyIterator
from prob_transformer.data.rna_handler import RNAHandler
from prob_transformer.data.ssd_handler import SSDHandler
from prob_transformer.data.mol_handler import MolHandler

from prob_transformer.module.geco_criterion import GECOLoss
from prob_transformer.routine.evaluation import run_evaluation
from prob_transformer.routine.training import run_epoch

device = 0 if torch.cuda.is_available() else "cpu"


def train_prob_transformer(config):
    expt_dir = pathlib.Path("experiment")
    sup = Supporter(experiments_dir=expt_dir, config_dict=config)

    cfg = sup.get_config()
    log = sup.get_logger()
    ckp = sup.ckp
    log.print_config(cfg)

    rank = 0

    np.random.seed(cfg.train.seed + rank)
    torch.manual_seed(cfg.train.seed + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed(cfg.train.seed + rank)
    else:
        rank = 'cpu'

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    log.log(f"rank {rank} ### START TRAINING ### at {socket.gethostname()}")

    ############################################################
    #######               DATA ITERATOR                 ########
    ############################################################
    log.log(f"### load data", rank=rank)

    num_props = False

    if cfg.data.type == "rna":

        ignore_index = -1
        pad_index = 0

        train_data = cinit(RNAHandler, cfg.data.rna, sub_set='train', prob_training="prob" in cfg.model.model_type,
                           device=rank, seed=cfg.data.seed, ignore_index=ignore_index)

        valid_data = cinit(RNAHandler, cfg.data.rna, sub_set='valid', prob_training=False, device=rank,
                           seed=cfg.data.seed, ignore_index=ignore_index)

        test_data = cinit(RNAHandler, cfg.data.rna, sub_set='test', prob_training=False, device=rank,
                          seed=cfg.data.seed, ignore_index=ignore_index)

        seq_vocab_size = train_data.seq_vocab_size
        trg_vocab_size = train_data.struct_vocab_size

    elif cfg.data.type == "ssd":

        ignore_index = -1
        pad_index = 0

        train_data = cinit(SSDHandler, cfg.data.ssd, sample_amount=cfg.data.ssd.sample_amount, device=rank,
                           pre_src_vocab=None, pre_trg_vocab=None, token_dict=None)
        valid_data = cinit(SSDHandler, cfg.data.ssd, sample_amount=cfg.data.ssd.sample_amount // 10, device=rank,
                           pre_src_vocab=train_data.pre_src_vocab, pre_trg_vocab=train_data.pre_trg_vocab,
                           token_dict=train_data.token_dict, seed=cfg.data.ssd.seed+1)
        test_data = cinit(SSDHandler, cfg.data.ssd, sample_amount=cfg.data.ssd.sample_amount // 10, device=rank,
                          pre_src_vocab=train_data.pre_src_vocab, pre_trg_vocab=train_data.pre_trg_vocab,
                          token_dict=train_data.token_dict, seed=cfg.data.ssd.seed+2)

        seq_vocab_size = train_data.source_vocab_size
        trg_vocab_size = train_data.target_vocab_size


    elif cfg.data.type == "mol":

        train_data = cinit(MolHandler, cfg.data.mol, split="train", device=rank)
        valid_data = cinit(MolHandler, cfg.data.mol, split="valid", device=rank)
        test_data = cinit(MolHandler, cfg.data.mol, split="test", device=rank)

        ignore_index = -1
        pad_index = train_data.ignore_index

        if isinstance(cfg.data.mol.props, List):
            num_props = len(cfg.data.mol.props)

        if "decoder" in cfg.model.model_type:
            seq_vocab_size = train_data.target_vocab_size
        else:
            seq_vocab_size = 1
        trg_vocab_size = train_data.target_vocab_size
        log(f"trg_vocab_size: {trg_vocab_size}")

    else:
        raise UserWarning(f"data type unknown: {cfg.data.type}")

    log.log(f"### load iterator", rank=rank)

    train_iter = MyIterator(data_handler=train_data, batch_size=cfg.data.batch_size, repeat=True, shuffle=True,
                            batching=True, pre_sort_samples=True,
                            device=rank, seed=cfg.data.seed + rank, ignore_index=ignore_index, pad_index=pad_index)

    valid_iter = MyIterator(data_handler=valid_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                            batching=True, pre_sort_samples=False,
                            device=rank, seed=cfg.data.seed + rank, ignore_index=ignore_index, pad_index=pad_index)

    test_iter = MyIterator(data_handler=test_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                           batching=False, pre_sort_samples=False,
                           device=rank, seed=cfg.data.seed + rank, ignore_index=ignore_index, pad_index=pad_index)

    log.log("train_set_size", train_iter.set_size, rank=rank)
    log.log("valid_set_size", valid_iter.set_size, rank=rank)
    log.log("test_set_size", test_iter.set_size, rank=rank)

    log("src_vocab_len", seq_vocab_size)
    log("tgt_vocab_len", trg_vocab_size)

    ############################################################
    #######                 BUILD MODEL                 ########
    ############################################################
    model = cinit(ProbTransformer, cfg.model, seq_vocab_size=seq_vocab_size, trg_vocab_size=trg_vocab_size,
                  props=num_props)

    log.log("model_parameters", count_parameters(model.parameters()), rank=rank)

    model = model.to(rank)

    train_summary = SummaryDict()
    eval_summary = SummaryDict()
    start_epoch = 0

    if "prob" in cfg.model.model_type:
        geco_criterion = cinit(GECOLoss, cfg.geco_criterion, model=model).to(rank)
    else:
        geco_criterion = None

    log.log(f"rank {rank} start GPU training ")

    log.log("trainable_parameters", count_parameters(model.parameters()), rank=rank)

    optima = cinit(OptiMaster, cfg.optim, model=model, epochs=cfg.train.epochs, iter_per_epoch=cfg.train.iter_per_epoch)

    ############################################################
    #######                START TRAINING               ########
    ############################################################
    log.start_timer(f"total", rank=rank)
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        log.start_timer(f"epoch", rank=rank)

        log(f"#{rank}## START epoch {epoch} " + '#' * 36)

        if epoch != 0:  # validate untrained model before start train
            log.start_timer(f"train", rank=rank)
            log("## Start training")
            stats, summary = run_epoch(rank, cfg, train_iter, model, geco_criterion=geco_criterion, opti=optima)
            for name, value in stats.items():
                log(f'train_' + name, value, epoch, rank=rank)

            train_summary(summary)
            eval_summary["step_count"] = stats['step']
            eval_summary["step"] = epoch
            eval_summary["train_loss"] = stats['loss']
            log.timer(f"train", epoch, rank=rank)

            ### Update Kappa
            if cfg.geco_criterion.kappa_adaption and model.probabilistic:
                kappa = geco_criterion.kappa.data.item()
                if summary["rec_constraint"].mean() < 0 and summary["lamb"].mean() < 1:
                    kappa = kappa + summary["rec_constraint"].mean()
                    log(f"kappa_update: rc mean {summary['rec_constraint'].mean():4.3f} new_kappa {kappa:6.6f}",
                        rank=rank)
                    geco_criterion.kappa.data = torch.FloatTensor([kappa]).to(rank)
        else:
            eval_summary["step_count"] = 0
            eval_summary["step"] = epoch
            eval_summary["train_loss"] = 0

        optima.epoch_step(epoch - 1)

        if model.probabilistic:
            eval_summary["kappa"] = geco_criterion.kappa
        eval_summary['learning_rate'] = optima.lr

        if cfg.data.type != 'mol':
            log.start_timer(f"valid", rank=rank)
            log("## Start validation")
            stats, summary = run_epoch(rank, cfg, valid_iter, model)
            for name, value in stats.items(): log(f'valid_' + name, value, epoch, rank=rank)
            eval_summary["valid_loss"] = stats['loss']
            log.timer(f"valid", epoch, rank=rank)

        if epoch % cfg.train.eval_freq == 0 and (rank == 0 or rank == 'cpu'):
            log.start_timer(f"eval", rank=rank)
            log("## Start valid evaluation")
            score_dict_valid = run_evaluation(cfg, valid_iter, model)
            for name, score in score_dict_valid.items():
                log(f"{name}_valid", score, epoch, rank=rank)
                eval_summary[f"{name}_valid"] = score
            log.timer(f"eval", epoch, rank=rank)

        train_summary.save(ckp.dir / "train_summary.npy")
        eval_summary.save(ckp.dir / "eval_summary.npy")
        log.save_to_json(rank=rank)

        log.timer(f"epoch", epoch, rank=rank)
        if cfg.data.type != 'mol':
            if stats['loss'] == np.nan:
                log(f"### STOP TRAINING - loss is NaN -> {stats['loss']}", rank=rank)
                break

        if epoch % cfg.train.save_freq == 0 and epoch != 0 and rank == 0 and cfg.expt.save_model:
            log(f"#{rank}## Save Model - number {epoch}", rank=rank)

            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optima.optimizer.state_dict(),
                          "config": cfg.get_dict}
            torch.save(checkpoint, ckp.dir / f"checkpoint_{epoch}.pth")

    ###########################################################
    ######                  TEST MODEL                 ########
    ###########################################################
    if rank == 0 or rank == 'cpu':
        log(f"#{rank}## FINAL TEST MODEL")
        log.start_timer(f"test")
        score_dict = run_evaluation(cfg, test_iter, model)  # , threshold=score_dict_valid['threshold'])

        for name, score in score_dict.items():
            log(f"{name}_test", score, epoch)
            eval_summary[f"{name}_final"] = score
        log.timer(f"test", epoch)
        log.save_to_json(rank=rank)
        eval_summary.save(ckp.dir / "eval_summary.npy")

    if cfg.expt.save_model:
        log(f"#{rank}## Save Model - final", rank=rank)

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optima.optimizer.state_dict(),
                      "config": cfg.get_dict}
        torch.save(checkpoint, ckp.dir / f"checkpoint_final.pth")

    log.timer(f"total", rank=rank)
    log(f"#{rank}## END TRAINING")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Train the model as specified by the given configuration file.')
    parser.add_argument('-c', '--config', type=str, help='a configuration file')
    args = parser.parse_args()

    if args.config:
        config = yaml.load(open(pathlib.Path.cwd() / pathlib.Path(args.config)), Loader=yaml.Loader)
    else:

        config = {
            "expt": {
                "experiment_name": "test_training",
                "save_model": True,
            },
            "train": {
                "eval_freq": 1,  # every * epoch will the model be evaluated
                "save_freq": 1,
                "seed": 1,  # random seed of numpy and torch
                "epochs": 10,  # epoch to train
                "n_sampling": 1,  # use dropout sampling during inference
                "iter_per_epoch": 100,  # number of samples drawn during evaluation
                "amp": False,  # automatic mixed precision
                "grad_scale": 2 ** 16,  # 2**16
            },
            "geco_criterion": {
                "kappa": 0.1,
                "kappa_adaption": True,
                "lagmul_rate": 0.01,
                "ma_decay": 0.99,
            },
            "model": {
                "model_type": 'prob_encoder',
                # "model_type": 'prob_decoder',
                "model_dim": 256,  # hidden dimension of transformer
                "max_len": 100,  # Maximum length an input sequence can have. Required for positional encoding.
                "n_layers": 4,  # number of transformer layers
                "num_head": 4,  # number of heads per layer
                "ff_factor": 4,  # hidden dim * ff_factor = size of feed-forward layer
                "z_factor": 1.0,  # hidden dim * z_factor = size of prob layer
                "dropout": 0.1,
                "prob_layer": "all",  # "middle", # middle all
                "zero_init": True,  # init last layer per block before each residual connection
            },
            "optim": {
                "optimizer": "adam",  # adam adamW rmsprop adabelief
                "scheduler": "cosine",  # cosine linear
                "warmup_epochs": 1,
                "lr_low": 0.0001,
                "lr_high": 0.0005,
                "clip_grad": 100,
                "beta1": 0.9,
                "beta2": 0.98,
                "weight_decay": 1e-10,
            },
            "data": {
                "type": 'rna',  # rna, ssd, mol
                "batch_size": 500,
                "seed": 1,
                "rna": {
                    "df_path": 'data/rna_data.plk',
                    "df_set_name": 'train',
                    "min_length": 20,
                    "max_length": 100,
                    "similarity": 80,
                },
                "ssd": {
                    "min_len": 15,
                    "max_len": 30,
                    "sample_amount": 10_000,
                    "trg_vocab_size": 50,
                    "src_vocab_size": 50,
                    "sentence_len": 3,
                    "n_sentence": 100,
                    "sentence_variations": 10,
                    "seed": 100,
                    "n_eval": 10,
                },
                "mol": {
                    "data_dir": "data/guacamol2.csv",
                    "min_length": 10,
                    "max_length": 100,
                    'props': ["tpsa", "logp", "sas"],
                    "gen_size": 100,
                    "block_size": 100,
                    "seed": 1,
                },
            },
        }
    train_prob_transformer(config=config)
