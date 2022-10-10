from typing import List
import socket
import numpy as np
import torch
import random
from prob_transformer.utils.handler.config import ConfigHandler
from prob_transformer.utils.logger import Logger
from prob_transformer.utils.summary import SummaryDict
from prob_transformer.utils.config_init import cinit
from prob_transformer.utils.torch_utils import count_parameters

from prob_transformer.model.probtransformer import ProbTransformer
from prob_transformer.data.iterator import MyIterator
from prob_transformer.data.rna_handler import RNAHandler
from prob_transformer.data.ssd_handler import SSDHandler
from prob_transformer.data.mol_handler import MolHandler

from prob_transformer.routine.evaluation import run_evaluation


def eval_transformer(checkpoint):
    cfg = ConfigHandler(config_dict=checkpoint['config'])

    infer_seed = cfg.train.seed
    torch.manual_seed(infer_seed)
    random.seed(infer_seed)
    np.random.seed(infer_seed)

    log = Logger("experiment", file_name="eval_log_file.txt")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    rank = 0 if torch.cuda.is_available() else "cpu"

    log.log(f"### START EVALUATION ### at {socket.gethostname()}")

    ############################################################
    #######               DATA ITERATOR                 ########
    ############################################################
    log.log(f"### load data", rank=rank)

    num_props = False

    if cfg.data.type == "rna":

        ignore_index = -1
        pad_index = 0

        train_data = cinit(RNAHandler, cfg.data.rna, df_path='data/rna_data.plk',  sub_set='train', prob_training="prob" in cfg.model.model_type,
                           device=rank, seed=cfg.data.seed, ignore_index=ignore_index)

        valid_data = cinit(RNAHandler, cfg.data.rna, df_path='data/rna_data.plk', sub_set='valid', prob_training=False, device=rank,
                           seed=cfg.data.seed, ignore_index=ignore_index)

        test_data = cinit(RNAHandler, cfg.data.rna, df_path='data/rna_data.plk', sub_set='test', prob_training=False, device=rank,
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
                           token_dict=train_data.token_dict)
        test_data = cinit(SSDHandler, cfg.data.ssd, sample_amount=cfg.data.ssd.sample_amount // 10, device=rank,
                          pre_src_vocab=train_data.pre_src_vocab, pre_trg_vocab=train_data.pre_trg_vocab,
                          token_dict=train_data.token_dict)

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
    valid_iter = MyIterator(data_handler=valid_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                            batching=True, pre_sort_samples=False,
                            device=rank, seed=cfg.data.seed + rank, ignore_index=ignore_index, pad_index=pad_index)

    test_iter = MyIterator(data_handler=test_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                           batching=False, pre_sort_samples=False,
                           device=rank, seed=cfg.data.seed + rank, ignore_index=ignore_index, pad_index=pad_index)

    log.log("valid_set_size", valid_iter.set_size, rank=rank)
    log.log("test_set_size", test_iter.set_size, rank=rank)

    log("src_vocab_len", seq_vocab_size)
    log("tgt_vocab_len", trg_vocab_size)

    ############################################################
    #######                 BUILD MODEL                 ########
    ############################################################
    model = cinit(ProbTransformer, cfg.model, seq_vocab_size=seq_vocab_size, trg_vocab_size=trg_vocab_size,
                  props=num_props)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    log.log("model_parameters", count_parameters(model.parameters()), rank=rank)

    model = model.to(rank)

    ############################################################
    #######                START EVALUATION             ########
    ############################################################
    eval_summary = SummaryDict()
    log.start_timer(f"eval", rank=rank)
    log("## Start valid evaluation")
    score_dict_valid = run_evaluation(cfg, valid_iter, model)
    for name, score in score_dict_valid.items():
        log(f"{name}_valid", score, rank=rank)
        eval_summary[f"{name}_valid"] = score
    log.timer(f"eval", rank=rank)

    ###########################################################
    ######                  TEST MODEL                 ########
    ###########################################################
    log.start_timer(f"test")
    score_dict_test = run_evaluation(cfg, test_iter, model, threshold=score_dict_valid['threshold'])
    for name, score in score_dict_test.items():
        log(f"{name}_test", score)
        eval_summary[f"{name}_test"] = score
    log.timer(f"test")
    log.save_to_json(rank=rank)
    eval_summary.save(cfg.expt.experiment_dir / "evaluation.npy")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the model given a checkpoint file.')
    parser.add_argument('-c', '--checkpoint', type=str, help='a checkpoint file')

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint,
                            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    eval_transformer(checkpoint=checkpoint)
