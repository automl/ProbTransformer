import pathlib

from tqdm import tqdm
import numpy as np
import torch

from prob_transformer.utils.torch_utils import count_parameters
from prob_transformer.utils.config_init import cinit
from prob_transformer.utils.handler.config import ConfigHandler

from prob_transformer.module.mat_head import SimpleMatrixHead
from prob_transformer.data.rna_handler import RNAHandler
from prob_transformer.data.dummy_handler import DummyHandler
from prob_transformer.data.iterator import MyIterator
from prob_transformer.evaluation.statistics_center import StatisticsCenter
from prob_transformer.module.optim_builder import OptiMaster

from prob_transformer.utils.supporter import Supporter


def run_cnn_training(config, expt_dir):
    model_dir = pathlib.Path(config['model_dir'])
    model_file = config['model_file']

    sup = Supporter(experiments_dir=expt_dir, config_dict=config, count_expt=True)
    cfg = sup.get_config()
    log = sup.get_logger()

    ckp = sup.ckp

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    rank = 0 if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_dir / model_file,
                            map_location=torch.device('cuda', rank) if rank == 0 else torch.device('cpu'))

    cfg_ckp = ConfigHandler(config_dict=checkpoint['config'])

    ignore_index = -1

    train_data = torch.load(model_dir / f"model_inference_train.pth")
    valid_data = torch.load(model_dir / f"model_inference_valid.pth")
    test_data = torch.load(model_dir / f"model_inference_test.pth")

    train_data = DummyHandler(train_data, max_length=cfg.max_train_len, device=0, max_hamming=5)
    valid_data = DummyHandler(valid_data, max_length=500, device=0)
    test_data = DummyHandler(test_data, max_length=500, device=0)

    rna_data = cinit(RNAHandler, cfg_ckp.data.rna.dict, df_path='data/rna_data.plk',
                     sub_set='valid', prob_training=True, device='cpu', seed=cfg_ckp.data.seed,
                     ignore_index=ignore_index, max_length=500, similarity='80', exclude=[])

    train_iter = MyIterator(data_handler=train_data, batch_size=cfg.batch_size, repeat=True, shuffle=True,
                            batching=True, pre_sort_samples=False,
                            device=rank, seed=cfg_ckp.data.seed, ignore_index=ignore_index)

    valid_iter = MyIterator(data_handler=valid_data, batch_size=1000, repeat=False, shuffle=False,
                            batching=True, pre_sort_samples=False,
                            device=rank, seed=cfg_ckp.data.seed, ignore_index=ignore_index)

    test_iter = MyIterator(data_handler=test_data, batch_size=1000, repeat=False, shuffle=False,
                           batching=True, pre_sort_samples=False,
                           device=rank, seed=cfg_ckp.data.seed, ignore_index=ignore_index)

    mat_model = cinit(SimpleMatrixHead, cfg.model)
    mat_model = mat_model.to(rank)

    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-1, label_smoothing=False).to(rank)

    epochs = 20

    opti = cinit(OptiMaster, cfg.opti, model=mat_model,
                 epochs=epochs, iter_per_epoch=cfg.iter_per_epoch)

    log.log("train_set_size", train_iter.set_size)
    log.log("valid_set_size", valid_iter.set_size)
    log.log("test_set_size", test_iter.set_size)
    log.log("model_parameters", count_parameters(mat_model.parameters()))

    for e in range(epochs):

        log(f"Start training epoch: ", e)

        for i, batch in tqdm(enumerate(train_iter)):

            if i == cfg.iter_per_epoch:
                break

            pred_mat, mask = mat_model(latent=batch.raw_latent, src=batch.src_seq, pred=batch.pred_struct,
                                       src_len=batch.length[:, 0])

            mask = mask[:, 0, :, :]
            trg_mat = batch.true_mat

            pred_mat = pred_mat * mask[:, :, :, None]
            trg_mat = trg_mat * mask

            loss_1 = criterion(pred_mat.permute(0, 3, 1, 2), trg_mat.to(torch.long))

            loss = loss_1 * mask

            def top_k_masking(loss, src_len, k_percent):
                with torch.no_grad():
                    mask = torch.zeros_like(loss)
                    for b in range(loss.shape[0]):
                        k = max(2, int(src_len[b] ** 2 * (k_percent / 100)))
                        idx = torch.topk(loss[b].view(-1), k=k)[1]
                        mask[b].view(-1)[idx] = 1
                return loss * mask, mask

            loss, mask = top_k_masking(loss, batch.length[:, 0], k_percent=cfg.k_percent)

            loss = torch.sum(loss, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
            loss = loss.mean()

            if i % 1000 == 0:
                log(f"loss", loss)

            loss.backward()

            opti.optimizer.step()
            opti.train_step()
            opti.optimizer.zero_grad()

        torch.save(mat_model, ckp.dir / f"checkpoint_{e}.pth")

        mat_model.eval()

        evaluations_mat = []
        for batch in tqdm(valid_iter):

            with torch.no_grad():
                b_pred_mat, mask = mat_model(latent=batch.raw_latent, src=batch.src_seq, pred=batch.pred_struct,
                                             src_len=batch.length[:, 0])

            for b, l in enumerate(batch.length[:, 0]):
                pred_mat = torch.sigmoid(b_pred_mat[b, :l, :l, 1])
                true_mat = batch.true_mat[b, :l, :l]

                pred_mat = torch.triu(pred_mat, diagonal=1).t() + torch.triu(pred_mat, diagonal=1)

                sequence = [rna_data.seq_itos[i] for i in
                            batch.src_seq[b, :l].detach().cpu().numpy()]

                sample = {"true": true_mat.detach().cpu().numpy(), "pred": pred_mat.detach().cpu().numpy(),
                          "sequence": sequence, }
                evaluations_mat.append(sample)

        new_stats = StatisticsCenter(evaluations_mat, step_size=0.2, triangle_loss=False)
        metrics, threshold = new_stats.find_best_threshold()
        for key, value in metrics.items():
            log(f"valid_{key}", value)

        log(f"### test epoch ", e)
        mat_model.eval()

        evaluations_mat = []
        for batch in tqdm(test_iter):

            with torch.no_grad():
                b_pred_mat, mask = mat_model(latent=batch.raw_latent, src=batch.src_seq, pred=batch.pred_struct,
                                             src_len=batch.length[:, 0])

            for b, l in enumerate(batch.length[:, 0]):
                pred_mat = torch.sigmoid(b_pred_mat[b, :l, :l, 1])
                true_mat = batch.true_mat[b, :l, :l]

                pred_mat = torch.triu(pred_mat, diagonal=1).t() + torch.triu(pred_mat, diagonal=1)

                sequence = [rna_data.seq_itos[i] for i in
                            batch.src_seq[b, :l].detach().cpu().numpy()]

                sample = {"true": true_mat.detach().cpu().numpy(), "pred": pred_mat.detach().cpu().numpy(),
                          "sequence": sequence, }
                evaluations_mat.append(sample)

        new_stats = StatisticsCenter(evaluations_mat, step_size=0.2, triangle_loss=False)
        metrics, threshold = new_stats.find_best_threshold()
        for key, value in metrics.items():
            log(f"test_{key}", value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the model given a checkpoint file.')
    parser.add_argument('-d', '--model_dir', type=str, help='the directory of a model')
    parser.add_argument('-f', '--model_file', type=str, help='the checkpoint file')

    args = parser.parse_args()

    model_dir = args.model_dir
    model_file = args.model_file

    config = {}

    config['model_dir'] = model_dir
    config['model_file'] = model_file
    config['expt'] = {
        "project_name": "tmp_test",
        "session_name": "cnn_head",
        "experiment_name": "test_model",
        "job_name": "local_run",
        "save_model": False,
        "resume_training": False,
    }

    config['seed'] = 5786
    config['batch_size'] = 1000
    config['max_train_len'] = 200
    config['k_percent'] = 0.4
    config['iter_per_epoch'] = 1000
    config['data'] = {'type':'rna'}

    config['opti'] = {"optimizer": 'adamW',  # 'adam',
                      "scheduler": 'cosine',
                      "warmup_epochs": 1,  # 1,
                      "lr_low": 0.0001,
                      "lr_high": 0.001,
                      "beta1": 0.9,
                      "beta2": 0.98,  # 0.98,
                      "weight_decay": 1e-10,  # 1e-8,
                      "factor": 1,
                      "swa": False,
                      "swa_start_epoch": 0,
                      "swa_lr": 0,
                      "plateua_metric": None}

    config['model'] = {"src_vocab_size": 5,
                       "latent_dim": 512,
                       "dropout": 0.1,
                       "model_dim": 64,
                       "out_channels": 2,
                       "res_layer": 3,
                       "kernel": 5,
                       "max_len": 500, }

    run_cnn_training(config, "experiments")
