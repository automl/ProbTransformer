import argparse, os
import wget
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import distance
import sys

from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.utils.config_init import cinit
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.utils.handler.config import ConfigHandler
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.model.probtransformer import ProbTransformer
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.data.rna_handler import RNAHandler
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.data.iterator import MyIterator
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.evaluation.statistics_center import StatisticsCenter
from RnaBench.lib.rna_folding_algorithms.DL.ProbTransformer.prob_transformer.routine.evaluation import is_valid_structure,correct_invalid_structure, struct_to_mat

rng = np.random.default_rng(seed=0)

NUCS = {
    'T': 'U',
    'P': 'U',
    'R': 'A',  # or 'G'
    'Y': 'C',  # or 'T'
    'M': 'C',  # or 'A'
    'K': 'U',  # or 'G'
    'S': 'C',  # or 'G'
    'W': 'U',  # or 'A'
    'H': 'C',  # or 'A' or 'U'
    'B': 'U',  # or 'G' or 'C'
    'V': 'C',  # or 'G' or 'A'
    'D': 'A',  # or 'G' or 'U'
    'N': rng.choice(['A', 'C', 'G', 'U']),  # 'N',
    'A': 'A',
    'U': 'U',
    'C': 'C',
    'G': 'G',
}

class ProbabilisticTransformer():
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(here)
        
        self.rank = 'cpu'
        
        model_path="RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/prob_transformer_final.pth"
        cnn_head_path="RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/cnn_head_final.pth"
        # rna_data="RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/data/rna_data.plk"
        
        
        if cnn_head_path == "RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/cnn_head_final.pth" and not os.path.exists("RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/cnn_head_final.pth"):
            os.makedirs("checkpoints", exist_ok=True)
            print("Download CNN head checkpoint")
            wget.download("https://ml.informatik.uni-freiburg.de/research-artifacts/probtransformer/cnn_head_final.pth", "RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/cnn_head_final.pth")
        
        if model_path == "RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/prob_transformer_final.pth" and not os.path.exists("RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/prob_transformer_final.pth"):
            os.makedirs("checkpoints", exist_ok=True)
            print("Download prob transformer checkpoint")
            wget.download("https://ml.informatik.uni-freiburg.de/research-artifacts/probtransformer/prob_transformer_final.pth", "RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/checkpoints/prob_transformer_final.pth")
        
        
        transformer_checkpoint = torch.load(model_path, map_location=torch.device(self.rank))
        
        
        cfg = ConfigHandler(config_dict=transformer_checkpoint['config'])
        
        # self.rna_data = cinit(RNAHandler, cfg.data.rna.dict, df_path=rna_data, sub_set='valid', prob_training=True,
        #                    device=self.rank, seed=cfg.data.seed, ignore_index=-1, similarity='80', exclude=[], max_length=500)

        self.seq_vocab = ['A', 'C', 'G', 'U', 'N']
        self.struct_vocab = ['.', '(0c', ')0c', '(1c', ')1c', '(2c', ')2c', '(0nc', ')0nc', '(1nc', ')1nc', '(2nc',
                             ')2nc']
        
        self.seq_stoi = dict(zip(self.seq_vocab, range(len(self.seq_vocab))))
        self.seq_itos = dict((y, x) for x, y in self.seq_stoi.items())

        for nuc, mapping in NUCS.items():
            self.seq_stoi[nuc] = self.seq_stoi[mapping]

        self.struct_itos = dict(zip(range(len(self.struct_vocab)), self.struct_vocab))
        self.struct_stoi = dict((y, x) for x, y in self.struct_itos.items())

        self.seq_vocab_size = len(self.seq_vocab)
        self.struct_vocab_size = len(self.struct_vocab)        
                
        self.model = cinit(ProbTransformer, cfg.model.dict, seq_vocab_size=self.seq_vocab_size, trg_vocab_size=self.struct_vocab_size,
                      mat_config=None, mat_head=False, mat_input=False, prob_ff=False,
                      scaffold=False, props=False).to(self.rank)
        
        self.model.load_state_dict(transformer_checkpoint['state_dict'], strict=False)
        
        self.cnn_head = torch.load(cnn_head_path, map_location=torch.device(self.rank))

    def __name__(self):
        return 'ProbTransformer'

    def __repr__(self):
        return 'ProbTransformer'

    def __call__(self, sequence, id=0):
        
        self.cnn_head.eval()
        self.model.eval()

        if sequence is not None:
            # print(sequence)
            # print(len(sequence))
            # print(f"Fold input sequence {sequence}")
            if sorted(set(sequence)) != ['A', 'C', 'G', 'U']:
                sequence = ''.join([NUCS[x] for x in sequence])
                # raise UserWarning(f"unknown symbols in sequence: {set(sequence).difference('A', 'C', 'G', 'U')}. Please only use ACGU")
    
            src_seq = torch.LongTensor([[self.seq_stoi[s] for s in sequence]]).to(self.rank)
            src_len = torch.LongTensor([src_seq.shape[1]]).to(self.rank)
    
            raw_output, raw_latent = self.model(src_seq, src_len, infer_mean=True, output_latent=True)
    
            pred_dist = torch.nn.functional.one_hot(torch.argmax(raw_output, dim=-1),
                                                    num_classes=raw_output.shape[-1]).to(torch.float).detach()
            pred_token = torch.argmax(raw_output, dim=-1).detach()
    
            b_pred_mat, mask = self.cnn_head(latent=raw_latent, src=src_seq, pred=pred_token, src_len=src_len)
    
            pred_dist = pred_dist[0, :, :].detach().cpu()
            pred_argmax = torch.argmax(pred_dist, keepdim=False, dim=-1).numpy().tolist()
            pred_struct = [self.struct_itos[i] for i in pred_argmax]
            # print("Predicted structure without CNN head:", pred_struct)
            if not is_valid_structure(pred_struct):
                pred_struct = correct_invalid_structure(pred_struct, pred_dist, self.struct_stoi, src_seq.shape[1])
                print("correction pred_struct", pred_struct)
    
            pred_mat = torch.sigmoid(b_pred_mat[0, :, :, 1])
            pred_mat = torch.triu(pred_mat, diagonal=1).t() + torch.triu(pred_mat, diagonal=1)
            bindings_idx = np.where(pred_mat.cpu().detach().numpy() > 0.5)
            # print("Predicted binding from CNN head, open :", bindings_idx[0].tolist())
            # print("Predicted binding from CNN head, close:", bindings_idx[1].tolist())
            # print(max(bindings_idx[0]))
            return [[o, c, 0] for o, c in zip(bindings_idx[0].tolist(), bindings_idx[1].tolist())]  # add 0 to pairlist to be able to compute metrics






# i __name__ == "__main__":
# def eval_probtransformer(sequence):
    # parser = argparse.ArgumentParser(description='Using the ProbTransformer to fold an RNA sequence.')
    # parser.add_argument('-s', '--sequence', type=str, help='A RNA sequence as ACGU-string')
    # parser.add_argument('-m', '--model', default="checkpoints/prob_transformer_final.pth", type=str,
    #                     help='A checkpoint file for the model to use')
    # parser.add_argument('-c', '--cnn_head', default="checkpoints/cnn_head_final.pth", type=str,
    #                     help='A RNA sequence as ACGU-string')
    # parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluates model on the test set TS0')
    # parser.add_argument('-d', '--rna_data', default="data/rna_data.plk", type=str, help='Path to rna dataframe')
    # parser.add_argument('-t', '--test_data', default="data/TS0.plk", type=str, help='Path to test dataframe TS0')
    # parser.add_argument('-r', '--rank', default="cuda", type=str, help='Device to infer the model, cuda or cpu')
    # args = parser.parse_args()

