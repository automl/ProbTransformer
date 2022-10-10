import numpy as np
import torch
import distance

def eval_toy_sample(n_pred_dist, true_dist, length):
    true_dist = true_dist.to(n_pred_dist[0].device)
    true_binary = true_dist > 0
    n_pred_binary = torch.zeros_like(n_pred_dist[0]).to(n_pred_dist[0].device)

    seq_solved_list, correct_symbols_list = [], []

    for pred_dist in n_pred_dist:

        eos_idx = length

        sample_pred_dist = pred_dist[:eos_idx, :]
        sample_pred_max_idx = torch.max(sample_pred_dist, dim=1, keepdim=True)[0]
        sample_pred_binary = sample_pred_dist.ge(sample_pred_max_idx)

        n_pred_binary[:eos_idx, :] += sample_pred_binary[:eos_idx, :]

        sample_true_binary = true_binary[:eos_idx, :]

        if eos_idx > sample_true_binary.shape[0]:
            correction_true = torch.zeros_like(sample_pred_binary).to(n_pred_dist[0].device)
            correction_true[:eos_idx, :] = sample_true_binary
            sample_true_binary = correction_true
        elif eos_idx < sample_true_binary.shape[0]:
            correction_true = torch.zeros_like(sample_true_binary).to(n_pred_dist[0].device)
            correction_true[:eos_idx, :] = sample_pred_binary
            sample_pred_binary = correction_true

        correct_symbols = sample_pred_binary.logical_and(sample_true_binary)
        correct_symbols = torch.sum(correct_symbols, dim=1, dtype=torch.float)
        mean_correct_symbols = torch.mean(correct_symbols)
        seq_solved = int(mean_correct_symbols.ge(1).cpu().detach().numpy())

        correct_symbols = mean_correct_symbols.cpu().detach().numpy()

        seq_solved_list.append(seq_solved)
        correct_symbols_list.append(correct_symbols)

    avg_pred_binary = n_pred_binary / n_pred_dist.shape[0]

    samplewise_kl = torch.sum(true_dist * torch.log(true_dist / (avg_pred_binary + 1e-32)), dim=1, keepdim=True)
    samplewise_kl = torch.mean(samplewise_kl).cpu().detach().numpy()

    mean_div = torch.mean(0.5 * torch.sum(torch.abs(true_dist - avg_pred_binary), dim=1, keepdim=True)).cpu().detach().numpy()

    levenshtein = []
    for t, p in zip(true_dist, avg_pred_binary):
        levenshtein.append(distance.levenshtein(torch.where(t>0)[0].cpu().detach().tolist(), torch.where(p>0)[0].cpu().detach().tolist()))

    options_per_symbol = torch.sum(true_binary, dim=1, keepdim=True)
    choices_per_symbol = torch.sum(avg_pred_binary.ge(0.01), dim=1, keepdim=True, dtype=torch.float)
    diversity = (choices_per_symbol * options_per_symbol) / (
            options_per_symbol * options_per_symbol + 1e-9)  # additional multiplication to zero out blanks
    diversity = torch.mean(diversity).cpu().detach().numpy()

    return {"seq_solved": np.mean(seq_solved_list), "correct_symbols": np.mean(correct_symbols_list),
                         "samplewise_kl": samplewise_kl, "diversity": diversity, "total_variation":mean_div,
            "levenshtein":np.mean(levenshtein) }


def eval_toy_task(batch_pred_dist, batch_src, trg_length, data_set):
    metrics_list = []
    batch_true_dist = data_set.get_batch_dist(batch_src)

    for n_pred_dist, true_dist, length in zip(batch_pred_dist, batch_true_dist, trg_length):
        metrics_list.append(eval_toy_sample(n_pred_dist, true_dist, length))
    return metrics_list
