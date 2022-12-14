import argparse, os
import wget
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import distance

from prob_transformer.utils.config_init import cinit
from prob_transformer.utils.handler.config import ConfigHandler
from prob_transformer.model.probtransformer import ProbTransformer
from prob_transformer.data.rna_handler import RNAHandler
from prob_transformer.data.iterator import MyIterator
from prob_transformer.evaluation.statistics_center import StatisticsCenter
from prob_transformer.routine.evaluation import is_valid_structure,correct_invalid_structure, struct_to_mat



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Using the ProbTransformer to fold an RNA sequence.')
    parser.add_argument('-s', '--sequence', type=str, help='A RNA sequence as ACGU-string')
    parser.add_argument('-m', '--model', default="checkpoints/prob_transformer_final.pth", type=str,
                        help='A checkpoint file for the model to use')
    parser.add_argument('-c', '--cnn_head', default="checkpoints/cnn_head_final.pth", type=str,
                        help='A RNA sequence as ACGU-string')
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluates model on the test set TS0')
    parser.add_argument('-d', '--rna_data', default="data/rna_data.plk", type=str, help='Path to rna dataframe')
    parser.add_argument('-t', '--test_data', default="data/TS0.plk", type=str, help='Path to test dataframe TS0')
    parser.add_argument('-r', '--rank', default="cuda", type=str, help='Device to infer the model, cuda or cpu')
    args = parser.parse_args()

    if args.cnn_head == "checkpoints/cnn_head_final.pth" and not os.path.exists("checkpoints/cnn_head_final.pth"):
        os.makedirs("checkpoints", exist_ok=True)
        print("Download CNN head checkpoint")
        wget.download("https://ml.informatik.uni-freiburg.de/research-artifacts/probtransformer/cnn_head_final.pth", "checkpoints/cnn_head_final.pth")

    if args.model == "checkpoints/prob_transformer_final.pth" and not os.path.exists("checkpoints/prob_transformer_final.pth"):
        os.makedirs("checkpoints", exist_ok=True)
        print("Download prob transformer checkpoint")
        wget.download("https://ml.informatik.uni-freiburg.de/research-artifacts/probtransformer/prob_transformer_final.pth", "checkpoints/prob_transformer_final.pth")


    transformer_checkpoint = torch.load(args.model, map_location=torch.device('cuda'))

    cfg = ConfigHandler(config_dict=transformer_checkpoint['config'])

    rna_data = cinit(RNAHandler, cfg.data.rna.dict, df_path=args.rna_data, sub_set='valid', prob_training=True,
                       device=args.rank, seed=cfg.data.seed, ignore_index=-1, similarity='80', exclude=[], max_length=500)

    seq_vocab_size = rna_data.seq_vocab_size
    trg_vocab_size = rna_data.struct_vocab_size

    model = cinit(ProbTransformer, cfg.model.dict, seq_vocab_size=seq_vocab_size, trg_vocab_size=trg_vocab_size,
                  mat_config=None, mat_head=False, mat_input=False, prob_ff=False,
                  scaffold=False, props=False).to(args.rank)
    model.load_state_dict(transformer_checkpoint['state_dict'], strict=False)
    model.eval()

    cnn_head = torch.load(args.cnn_head)
    cnn_head.eval()

    if args.sequence is not None:
        print(f"Fold input sequence {args.sequence}")
        if sorted(set(args.sequence)) != ['A', 'C', 'G', 'U']:
            raise UserWarning(f"unknown symbols in sequence: {set(args.sequence).difference('A', 'C', 'G', 'U')}. Please only use ACGU")

        src_seq = torch.LongTensor([[rna_data.seq_stoi[s] for s in args.sequence]]).to(args.rank)
        src_len = torch.LongTensor([src_seq.shape[1]]).to(args.rank)

        raw_output, raw_latent = model(src_seq, src_len, infer_mean=True, output_latent=True)

        pred_dist = torch.nn.functional.one_hot(torch.argmax(raw_output, dim=-1),
                                                num_classes=raw_output.shape[-1]).to(torch.float).detach()
        pred_token = torch.argmax(raw_output, dim=-1).detach()

        b_pred_mat, mask = cnn_head(latent=raw_latent, src=src_seq, pred=pred_token, src_len=src_len)

        pred_dist = pred_dist[0, :, :].detach().cpu()
        pred_argmax = torch.argmax(pred_dist, keepdim=False, dim=-1).numpy().tolist()
        pred_struct = [rna_data.struct_itos[i] for i in pred_argmax]
        print("Predicted structure without CNN head:", pred_struct)
        if not is_valid_structure(pred_struct):
            pred_struct = correct_invalid_structure(pred_struct, pred_dist, rna_data.struct_stoi, src_seq.shape[1])
            print("correction pred_struct", pred_struct)

        pred_mat = torch.sigmoid(b_pred_mat[0, :, :, 1])
        pred_mat = torch.triu(pred_mat, diagonal=1).t() + torch.triu(pred_mat, diagonal=1)
        bindings_idx = np.where(pred_mat.cpu().detach().numpy() > 0.5)
        print("Predicted binding from CNN head, open :", bindings_idx[0].tolist())
        print("Predicted binding from CNN head, close:", bindings_idx[1].tolist())

    if args.evaluate:
        print("Evaluate on TS0")
        test_data = cinit(RNAHandler, cfg.data.rna.dict, df_path=args.test_data, sub_set='test', prob_training=True,
                     device=args.rank, seed=cfg.data.seed, ignore_index=-1, similarity='80', exclude=[], max_length=500)

        data_iter = MyIterator(data_handler=test_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                               batching=False, pre_sort_samples=False,
                               device=args.rank, seed=cfg.data.seed, ignore_index=-1)

        samples_count = 0
        metrics = defaultdict(list)
        evaluations = []

        with torch.inference_mode():
            for i, batch in tqdm(enumerate(data_iter)):
                raw_output, raw_latent = model(batch.src_seq, batch.src_len, infer_mean=True, output_latent=True)

                pred_dist = torch.nn.functional.one_hot(torch.argmax(raw_output, dim=-1),
                                                        num_classes=raw_output.shape[-1]).to(torch.float).detach()
                pred_token = torch.argmax(raw_output, dim=-1).detach()

                b_pred_mat, mask = cnn_head(latent=raw_latent, src=batch.src_seq, pred=pred_token,
                                            src_len=batch.src_len)

                for b, length in enumerate(batch.src_len.detach().cpu().numpy()):

                    samples_count += 1
                    sequence = [test_data.seq_itos[i] for i in
                                batch.src_seq[b, :length].detach().cpu().numpy()]

                    pred_struct = pred_dist[b, :length, :].detach().cpu()
                    true_struct = batch.trg_seq[b, :length].detach().cpu().numpy()

                    pred_argmax = torch.argmax(pred_struct, keepdim=False, dim=-1).numpy()
                    np_remove_pos = np.where(true_struct == -1)[0]
                    np_pred_struct = np.delete(pred_argmax, np_remove_pos).tolist()
                    np_true_struct = np.delete(true_struct, np_remove_pos).tolist()
                    hamming = distance.hamming(np_pred_struct, np_true_struct)

                    word_errors = [r != h for (r, h) in zip(pred_argmax, true_struct)]
                    word_error_rate = sum(word_errors) / len(true_struct)

                    metrics[f"hamming_distance"].append(hamming)
                    metrics[f"word_error_rate"].append(word_error_rate)

                    true_mat = batch.trg_pair_mat[b]
                    true_mat = true_mat.detach().cpu().numpy()
                    true_mat = true_mat[:length, :length]

                    vocab = {k: i for i, k in enumerate(test_data.struct_vocab)}
                    db_struct = [test_data.struct_itos[i] for i in pred_argmax]
                    if is_valid_structure(db_struct):
                        seq_pred_mat, pairs = struct_to_mat(db_struct)
                    else:
                        correct_struct = correct_invalid_structure(db_struct, pred_dist[b, :length, :],
                                                                   test_data.struct_stoi, length)
                        if is_valid_structure(correct_struct):
                            seq_pred_mat, pairs = struct_to_mat(correct_struct)
                        else:
                            seq_pred_mat = np.zeros_like(true_mat)

                    metrics[f"seq_solved"].append(np.all(np.equal(true_mat, seq_pred_mat > 0.5)).astype(int))
                    pred_mat = torch.sigmoid(b_pred_mat[b, :length, :length, 1])
                    pred_mat = torch.triu(pred_mat, diagonal=1).t() + torch.triu(pred_mat, diagonal=1)
                    sample = {"true": true_mat, "pred": pred_mat.detach().cpu().numpy(), "sequence": sequence, }
                    evaluations.append(sample)

        metrics = {k: np.mean(v) for k, v in metrics.items()}
        stats = StatisticsCenter(evaluations, step_size=0.1, triangle_loss=False)
        metric = stats.eval_threshold(0.5)
        metrics.update({k: v for k, v in metric.items()})
        metrics['samples'] = samples_count

        for key, value in metrics.items():
            print(f"Evaluate {args.test_data} {key:20}: {value}")


