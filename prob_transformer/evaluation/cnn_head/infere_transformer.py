from tqdm import tqdm
import numpy as np
import torch
import distance

from prob_transformer.utils.config_init import cinit
from prob_transformer.utils.handler.config import ConfigHandler
from prob_transformer.model.probtransformer import ProbTransformer
from prob_transformer.data.rna_handler import RNAHandler
from prob_transformer.data.iterator import MyIterator


def infer_rna_transformer(checkpoint):
    rank = 0 if torch.cuda.is_available() else "cpu"
    cfg = ConfigHandler(config_dict=checkpoint['config'])

    ignore_index = -1
    train_data = cinit(RNAHandler, cfg.data.rna.dict, df_path='data/rna_data.plk',
                       sub_set='train', prob_training=True, device=rank, seed=cfg.data.seed, ignore_index=ignore_index,
                       similarity='80',
                       exclude=[], max_length=500)
    valid_data = cinit(RNAHandler, cfg.data.rna.dict, df_path='data/rna_data.plk',
                       sub_set='valid', prob_training=True, device=rank, seed=cfg.data.seed, ignore_index=ignore_index,
                       similarity='80',
                       exclude=[], max_length=500)
    test_data = cinit(RNAHandler, cfg.data.rna.dict, df_path='data/rna_data.plk',
                      sub_set='test', prob_training=True, device=rank, seed=cfg.data.seed, ignore_index=ignore_index,
                      similarity='80',
                      exclude=[], max_length=500)

    seq_vocab_size = valid_data.seq_vocab_size
    trg_vocab_size = valid_data.struct_vocab_size

    train_iter = MyIterator(data_handler=train_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=True,
                            batching=True, pre_sort_samples=True,
                            device=rank, seed=cfg.data.seed, ignore_index=ignore_index)

    valid_iter = MyIterator(data_handler=valid_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                            batching=True, pre_sort_samples=False,
                            device=rank, seed=cfg.data.seed, ignore_index=ignore_index)

    test_iter = MyIterator(data_handler=test_data, batch_size=cfg.data.batch_size, repeat=False, shuffle=False,
                           batching=False, pre_sort_samples=False,
                           device=rank, seed=cfg.data.seed, ignore_index=ignore_index)

    model = cinit(ProbTransformer, cfg.model.dict, seq_vocab_size=seq_vocab_size, trg_vocab_size=trg_vocab_size,
                  mat_config=None, mat_head=False, mat_input=False, prob_ff=False,
                  scaffold=False, props=False).to(rank)

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()

    data_iter = {'train': train_iter, 'valid': valid_iter, 'test': test_iter}

    for d_set, d_iter in data_iter.items():
        raw_output_dump = []
        with torch.inference_mode():
            for i, batch in tqdm(enumerate(d_iter)):

                model.eval()

                with torch.no_grad():
                    raw_output, raw_latent = model(batch.src_seq, batch.src_len, infer_mean=True, output_latent=True)

                    trg_dist = torch.nn.functional.one_hot(torch.argmax(raw_output, dim=-1),
                                                           num_classes=raw_output.shape[-1]).to(torch.float).detach()
                    trg_token = torch.argmax(raw_output, dim=-1).detach()

                for b, length in enumerate(batch.src_len.detach().cpu().numpy()):
                    sequence = [d_iter.data_handler.seq_itos[i] for i in
                                batch.src_seq[b, :length].detach().cpu().numpy()]

                    pred_struct = trg_dist[b, :length, :].detach().cpu()
                    true_struct = batch.trg_seq[b, :length].detach().cpu()

                    pred_argmax = torch.argmax(pred_struct, keepdim=False, dim=-1)
                    np_remove_pos = np.where(true_struct.numpy() == -1)[0]
                    np_pred_struct = np.delete(pred_argmax.numpy(), np_remove_pos).tolist()
                    np_true_struct = np.delete(true_struct.numpy(), np_remove_pos).tolist()
                    hamming = distance.hamming(np_pred_struct, np_true_struct)

                    raw_output_dump.append({
                        "raw_output": raw_output[b, :length, :].detach().cpu(),
                        "raw_latent": raw_latent[b, :length, :].detach().cpu(),
                        "pred_struct": trg_token[b, :length].detach().cpu(),
                        "true_struct": true_struct,
                        "true_mat": batch.trg_pair_mat[b, :length, :length].detach().cpu(),
                        "src_seq": batch.src_seq[b, :length].detach().cpu(),
                        "trg_token": trg_token[b, :length].detach().cpu(),
                        "sequence": sequence,
                        "hamming": hamming,
                        "length": length
                    })

        torch.save(raw_output_dump, cfg.expt.experiment_dir / f"model_inference_{d_set}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the model given a checkpoint file.')
    parser.add_argument('-c', '--checkpoint', type=str, help='a checkpoint file')

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint,
                            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    infer_rna_transformer(checkpoint=checkpoint)
