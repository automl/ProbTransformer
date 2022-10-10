from collections import deque, defaultdict
import math
import re
import numpy as np
import distance
import torch
import collections
from tqdm import tqdm
import pandas as pd
from moses.utils import get_mol
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # disable rdkit error log
from prob_transformer.evaluation.metrics.sascore import calculateScore
from prob_transformer.data.mol_handler import check_novelty, sample_batch, canonic_smiles
from prob_transformer.evaluation.statistics_center import StatisticsCenter
from prob_transformer.evaluation.metrics.toy_task_survey import eval_toy_sample


def infer_encoder(model, src, src_len, n=1, dropout_sampling=False):
    pred_dist_list = []

    if n > 1 and dropout_sampling:
        model.train()
    else:
        model.eval()

    with torch.no_grad():
        if n == 1:
            raw_output = model(src, src_len, infer_mean=True).detach()
            trg_dist = torch.nn.functional.one_hot(torch.argmax(raw_output, dim=-1),
                                                   num_classes=raw_output.shape[-1]).to(torch.float)
            pred_dist_list.append(trg_dist)
        else:
            for _ in range(n):
                raw_output = model(src, src_len).detach()
                if model.probabilistic:
                    trg_dist = torch.nn.functional.one_hot(torch.argmax(raw_output, dim=-1),
                                                           num_classes=raw_output.shape[-1]).to(torch.float)
                else:
                    trg_dist = torch.softmax(raw_output, dim=-1)
                    samples = trg_dist.view(-1, trg_dist.shape[-1]).multinomial(1, True).view(trg_dist.shape[:-1])
                    trg_dist = torch.nn.functional.one_hot(samples, num_classes=raw_output.shape[-1]).to(torch.float)
                pred_dist_list.append(trg_dist)

    pred_raw_dist = torch.stack(pred_dist_list, 1)
    pred_mean = pred_raw_dist.mean(dim=1)
    pred_std = pred_raw_dist.std(dim=1)
    return pred_mean, pred_std, pred_raw_dist


def struct_to_mat(struct):
    bracket_buffer = {"0": deque(), "1": deque(), "2": deque()}
    struct_mat = np.zeros([len(struct), len(struct)])

    pairs = []

    for idx, symb in enumerate(struct):
        if '.' == symb[0]:
            continue
        else:
            if 'C' == symb[1]:
                continue
            else:
                if "(" == symb[0]:
                    bracket_buffer[symb[1]].append(idx)
                elif ")" == symb[0]:
                    if len(bracket_buffer[symb[1]]) == 0:
                        continue  # return False
                    open_idx = bracket_buffer[symb[1]].pop()

                    struct_mat[open_idx, idx] = 1
                    struct_mat[idx, open_idx] = 1
                    pairs.append((idx, open_idx))
    return struct_mat, pairs


def is_valid_structure(hypo):
    opener = ['(', '[', '<']
    closer = [')', ']', '>']
    valid_pairs = all([hypo.count(opener[l]) == hypo.count(closer[l]) for l in range(len(opener))])
    return valid_pairs


def get_unbalanced_brackets(hypo):
    # opening brackets
    op_l0 = []
    op_l1 = []
    op_l2 = []
    op_l3 = []

    # closing brackets
    cl_l0 = []
    cl_l1 = []
    cl_l2 = []
    cl_l3 = []
    for index, character in enumerate(hypo):
        if character == '(BP' or character == '(':
            op_l0.append((index, character))
        elif character == '(PK' or character == '[':
            op_l1.append((index, character))
        elif character == '(PK' or character == '{':
            op_l2.append((index, character))
        elif character == '(PK' or character == '<':
            op_l3.append((index, character))
        else:
            pass

        if character == ')BP' or character == ')':
            if not op_l0:
                cl_l0.append((index, character))
            else:
                op_l0.pop()
        elif character == ')PK' or character == ']':
            if not op_l1:
                cl_l1.append((index, character))
            else:
                op_l1.pop()
        elif character == ')PK' or character == '}':
            if not op_l2:
                cl_l2.append((index, character))
            else:
                op_l2.pop()
        elif character == ')PK' or character == '>':
            if not op_l3:
                cl_l3.append((index, character))
            else:
                op_l3.pop()
        else:
            pass
    return op_l0, op_l1, op_l2, op_l3, cl_l0, cl_l1, cl_l2, cl_l3


def unbalanced_brackets2dots(hypo):
    unbalanced_brackets = []

    for level in get_unbalanced_brackets(hypo):
        unbalanced_brackets += level

    for index, _ in unbalanced_brackets:
        hypo[index] = "."
    return hypo


def correct_invalid_structure(hypo, dist, stoi, length):
    opener = ['(', '[', ]
    closer = [')', ']', ]

    dist = dist.squeeze().cpu().numpy()

    if dist.shape[0] < length:
        dist_ = np.zeros([length, dist.shape[1]])
        dist_[:dist.shape[0], :] = dist
        dist = dist

    while len(hypo) < length:
        hypo.append('.')

    if len(hypo) > length:
        hypo = hypo[:length]
        dist = dist[:length, :]

    dist = np.exp(dist - np.max(dist, axis=1, keepdims=True)) / (
        np.sum(np.exp(dist - np.max(dist, axis=1, keepdims=True)), axis=1, keepdims=True))  # softmax

    for op, cl in zip(opener, closer):
        if (op in hypo or cl in hypo) and np.abs(hypo.count(op) - hypo.count(cl)) <= hypo.count("."):

            if hypo.count(op) < hypo.count(cl):
                for _ in range(hypo.count(cl) - hypo.count(op)):
                    symb_dist = dist[:, stoi[op]].copy()

                    brackets = np.where(np.asarray(hypo) != ".")[0]
                    symb_dist[brackets] = 0  # only "." positions

                    cl_pos = np.where(np.asarray(hypo) == cl)[0]
                    symb_dist[cl_pos[-1]:] = 0  # only positions before last closing bracket

                    change_idx = np.argmax(symb_dist)
                    change_idx = min(change_idx, len(hypo) - 1)
                    hypo[change_idx] = op

            if hypo.count(op) > hypo.count(cl):
                for _ in range(hypo.count(op) - hypo.count(cl)):
                    symb_dist = dist[:, stoi[cl]].copy()

                    brackets = np.where(np.asarray(hypo) != ".")[0]
                    symb_dist[brackets] = 0

                    op_pos = np.where(np.asarray(hypo) == op)[0]
                    symb_dist[:op_pos[0]] = 0

                    change_idx = np.argmax(symb_dist)
                    change_idx = min(change_idx, len(hypo) - 1)
                    hypo[change_idx] = cl

    if not is_valid_structure(hypo):
        hypo = unbalanced_brackets2dots(hypo)
    return hypo


def struct_to_mat(struct):
    bracket_buffer = {"0": deque(), "1": deque(), "2": deque()}
    struct_mat = np.zeros([len(struct), len(struct)])

    pairs = []

    for idx, symb in enumerate(struct):
        if '.' == symb[0]:
            continue
        else:
            if 'C' == symb[1]:
                continue
            else:
                if "(" == symb[0]:
                    bracket_buffer[symb[1]].append(idx)
                elif ")" == symb[0]:
                    if len(bracket_buffer[symb[1]]) == 0:
                        continue  # return False
                    open_idx = bracket_buffer[symb[1]].pop()

                    struct_mat[open_idx, idx] = 1
                    struct_mat[idx, open_idx] = 1
                    pairs.append((idx, open_idx))
    return struct_mat, pairs


def run_evaluation(cfg, data_iter, model, threshold=None):
    model.eval()

    if cfg.data.type == 'rna':

        metrics = defaultdict(list)

        samples_count = 0
        PDB_samples_count = 0

        evaluations_seq = []
        evaluations_mat = []

        with torch.inference_mode():
            for i, batch in enumerate(data_iter):

                pred_mean, pred_std, pred_raw_dist = infer_encoder(model, batch.src_seq, batch.src_len, n=1,
                                                                   dropout_sampling=False)

                for b, length in enumerate(batch.src_len.detach().cpu().numpy()):

                    samples_count += 1

                    pred_struct = torch.argmax(pred_mean[b, :length, :], keepdim=False, dim=-1).detach().cpu().numpy()
                    true_struct = batch.trg_seq[b, :length].detach().cpu().numpy()

                    remove_pos = np.where(true_struct == -1)[0]
                    pred_struct = np.delete(pred_struct, remove_pos).tolist()
                    true_struct = np.delete(true_struct, remove_pos).tolist()

                    hamming = distance.hamming(pred_struct, true_struct)
                    word_errors = [r != h for (r, h) in zip(pred_struct, true_struct)]
                    word_error_rate = sum(word_errors) / len(true_struct)

                    metrics[f"hamming_distance"].append(hamming)
                    metrics[f"word_error_rate"].append(word_error_rate)

                    if batch.pdb_sample[b] == 1:
                        PDB_samples_count += 1

                    true_mat = batch.trg_pair_mat[b]
                    true_mat = true_mat.detach().cpu().numpy()
                    true_mat = true_mat[:length, :length]

                    sequence = [data_iter.data_handler.seq_itos[i] for i in
                                batch.src_seq[b, :length].detach().cpu().numpy()]

                    sample_pred_mean = pred_mean[b, :length]
                    sample_pred_mean = sample_pred_mean / torch.sum(sample_pred_mean, dim=-1, keepdim=True)

                    vocab = {k: i for i, k in enumerate(data_iter.data_handler.struct_vocab)}

                    def mean_mat(pred_mean, s1, s2, vocab, dot=False):
                        vec_1 = pred_mean[:, vocab[s1]]
                        vec_2 = pred_mean[:, vocab[s2]]

                        mat = torch.matmul(vec_1.unsqueeze(1), vec_2.unsqueeze(0))
                        mat = mat.triu(1) + mat.triu(1).t()
                        if dot:
                            mat = mat + torch.eye(mat.shape[0]).to(mat.device)
                        return mat

                    mat_0c = mean_mat(sample_pred_mean, '(0c', ')0c', vocab)
                    mat_1c = mean_mat(sample_pred_mean, '(1c', ')1c', vocab)
                    mat_2c = mean_mat(sample_pred_mean, '(2c', ')2c', vocab)
                    mat_0nc = mean_mat(sample_pred_mean, '(0nc', ')0nc', vocab)
                    mat_1nc = mean_mat(sample_pred_mean, '(1nc', ')1nc', vocab)
                    mat_2nc = mean_mat(sample_pred_mean, '(2nc', ')2nc', vocab)

                    mat_c = mat_0c + mat_1c + mat_2c
                    mat_nc = mat_0nc + mat_1nc + mat_2nc

                    c_mask = torch.zeros_like(mat_c)
                    cn_mask = torch.zeros_like(mat_c)

                    for k in range(1, c_mask.shape[1] - 1):
                        for i, j in zip(range(0, c_mask.shape[1] - k), range(k, c_mask.shape[1])):
                            if ''.join([sequence[j], sequence[i]]) in data_iter.data_handler.canonical_pairs:
                                c_mask[i, j] = 1
                            else:
                                cn_mask[i, j] = 1

                    mat_joint = mat_c * c_mask + mat_nc * cn_mask
                    mat_threshold = 0.1

                    for k in range(1, mat_joint.shape[1] - 1):
                        for i, j in zip(range(0, mat_joint.shape[1] - k), range(k, mat_joint.shape[1])):
                            if mat_joint[i, j] > mat_threshold:
                                mat_joint[i, :] = 0
                                mat_joint[:, j] = 0
                                mat_joint[i, j] = 1

                    mat_joint = mat_joint > mat_threshold
                    mat_joint = mat_joint + mat_joint.t()
                    pred_mat = mat_joint.to(torch.float).cpu().numpy()

                    metrics[f"mat_solved"].append(np.all(np.equal(true_mat, pred_mat > 0.5)).astype(int))

                    sample = {"true": true_mat, "pred": pred_mat, "sequence": sequence, }
                    evaluations_mat.append(sample)

                    db_struct = [data_iter.data_handler.struct_itos[i] for i in pred_struct]
                    if is_valid_structure(db_struct):
                        seq_pred_mat, pairs = struct_to_mat(db_struct)
                    else:
                        correct_struct = correct_invalid_structure(db_struct, pred_mean[b, :length, :],
                                                                   data_iter.data_handler.struct_stoi, length)
                        if is_valid_structure(correct_struct):
                            seq_pred_mat, pairs = struct_to_mat(correct_struct)
                        else:
                            seq_pred_mat = np.zeros_like(true_mat)

                    metrics[f"seq_solved"].append(np.all(np.equal(true_mat, seq_pred_mat > 0.5)).astype(int))

                    sample = {"true": true_mat, "pred": seq_pred_mat, "sequence": sequence, }
                    evaluations_seq.append(sample)

        metrics = {k: np.mean(v) for k, v in metrics.items()}

        if len(evaluations_mat) > 1:
            stats = StatisticsCenter(evaluations_mat, step_size=0.1, triangle_loss=False)
            if threshold == None:
                metric, threshold = stats.find_best_threshold()
            else:
                metric = stats.eval_threshold(threshold)
            metrics.update({f"mat_{k}": v for k, v in metric.items()})

        if len(evaluations_seq) > 1:
            stats = StatisticsCenter(evaluations_seq, step_size=0.1, triangle_loss=False)
            if threshold == None:
                metric, threshold = stats.find_best_threshold()
            else:
                metric = stats.eval_threshold(threshold)
            metrics.update({f"seq_{k}": v for k, v in metric.items()})

        metrics['threshold'] = threshold

        metrics['samples'] = samples_count
        metrics['PDB_samples'] = PDB_samples_count

        return metrics

    elif cfg.data.type == 'ssd':

        samples_count = 0
        metrics_list = []

        dropout_sampling = not model.probabilistic

        with torch.inference_mode():
            for i, batch in enumerate(data_iter):

                pred_mean, pred_std, pred_raw_dist = infer_encoder(model, batch.src_seq, batch.src_len,
                                                                   n=cfg.data.ssd.n_eval,
                                                                   dropout_sampling=dropout_sampling)

                for b, length in enumerate(batch.src_len.detach().cpu().numpy()):

                    pred_dist = pred_raw_dist[b, :, :length, :]

                    if pred_dist.shape[1] < length:
                        pred_dist_tmp = torch.zeros([pred_dist.shape[0], length, pred_dist.shape[2]])
                        pred_dist_tmp[:, :pred_dist.shape[1], :] = pred_dist
                        pred_dist = pred_dist_tmp

                    samples_count += 1

                    true_source = batch.src_seq[b, :length].detach().cpu().numpy()
                    true_dist = data_iter.data_handler.get_sample_dist(true_source)
                    true_dist = torch.FloatTensor(true_dist)

                    metrics = eval_toy_sample(pred_dist, true_dist, length)
                    metrics_list.append(metrics)

        metrics = defaultdict(list)

        for sm in metrics_list:
            for k, v in sm.items():
                metrics[k].append(v)

        metrics = {k: np.mean(v) for k, v in metrics.items()}
        metrics['samples'] = samples_count
        metrics['threshold'] = threshold

        return metrics

    elif cfg.data.type == 'mol':

        prop2value = {'qed': [0.3, 0.5, 0.7], 'sas': [2.0, 3.0, 4.0], 'logp': [2.0, 4.0, 6.0],
                      'tpsa': [40.0, 80.0, 120.0],
                      'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0],
                                    [40.0, 6.0], [80.0, 6.0], [120.0, 6.0]],
                      'sas_logp': [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0],
                                   [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]],
                      'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 3.0], [80.0, 3.0], [120.0, 3.0],
                                   [40.0, 4.0], [80.0, 4.0], [120.0, 4.0]],
                      'tpsa_logp_sas': [[40.0, 2.0, 2.0], [40.0, 2.0, 4.0], [40.0, 6.0, 4.0], [40.0, 6.0, 2.0],
                                        [80.0, 6.0, 4.0], [80.0, 2.0, 4.0], [80.0, 2.0, 2.0], [80.0, 6.0, 2.0]]}

        chars = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=',
                 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]',
                 '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]',
                 '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]',
                 '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]',
                 '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]',
                 '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]',
                 'b', 'c', 'n', 'o', 'p', 's']

        prop_condition = [False]
        if cfg.data.mol.props:
            if len(cfg.data.mol.props) > 0:
                prop_condition = prop2value['_'.join(cfg.data.mol.props)]

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}

        dataset = data_iter.data_handler

        if dataset.split == "test":
            gen_size = 10000
        else:
            gen_size = 500

        batch_size = math.ceil(cfg.data.batch_size / cfg.data.mol.block_size)
        gen_iter = math.ceil(gen_size / batch_size)

        metrics = collections.defaultdict(list)

        count = 0

        for prop_c in prop_condition:
            molecules = []
            count += 1
            for i in tqdm(range(gen_iter)):
                x = torch.tensor([stoi[s] for s in regex.findall(dataset.context)], dtype=torch.long)[
                    None, ...].repeat(batch_size, 1).to('cuda')

                if prop_c:
                    p = torch.tensor([prop_c]).repeat(batch_size, 1).unsqueeze(1).to('cuda')  # for multiple conditions
                else:
                    p = None

                y = sample_batch(model, x, cfg.data.mol.block_size, temperature=1, sample=True, top_k=None, props=p,
                                 ignore_index=dataset.ignore_index)  # 0.7 for guacamol
                for gen_mol in y:
                    completion = ''.join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace('<', '')
                    mol = get_mol(completion)
                    if mol:
                        molecules.append(mol)

            if len(molecules) == 0:
                metrics["valid_ratio"].append(0)
                metrics["unique_ratio"].append(0)
                metrics["novelty_ratio"].append(0)
                if cfg.data.mol.props:
                    for prop_name in cfg.data.mol.props:
                        metrics[f'{prop_name}_MAD'].append(0)
                        metrics[f'{prop_name}_SD'].append(0)
                break

            mol_dict = []

            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

            results = pd.DataFrame(mol_dict)

            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))

            novel_ratio = check_novelty(unique_smiles, set(
                dataset.data[dataset.data['source'] == 'train']['smiles']))

            metrics["valid_ratio"].append(len(results) / (batch_size * gen_iter))
            metrics["unique_ratio"].append(len(unique_smiles) / len(results))
            metrics["novelty_ratio"].append(novel_ratio / 100)

            results['validity'] = np.round(len(results) / (batch_size * gen_iter), 3)
            results['unique'] = np.round(len(unique_smiles) / len(results), 3)
            results['novelty'] = np.round(novel_ratio / 100, 3)

            qed = results['molecule'].apply(lambda x: QED.qed(x))
            sas = results['molecule'].apply(lambda x: calculateScore(x))
            logp = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
            tpsa = results['molecule'].apply(lambda x: CalcTPSA(x))

            if cfg.data.mol.props:
                properties = {"qed": qed, "sas": sas, "logp": logp, "tpsa": tpsa}
                for true_value, prop_name in zip(prop_c, cfg.data.mol.props):
                    prop_values = properties[prop_name]

                    abs_div = np.abs(prop_values - true_value)

                    mad = np.mean(abs_div)
                    sd = np.std(abs_div)

                    metrics[f'{prop_name}_MAD'].append(mad)
                    metrics[f'{prop_name}_SD'].append(sd)

        rmetrics = {}
        for key, value_list in metrics.items():
            rmetrics[key] = np.mean(value_list)
        rmetrics['threshold'] = threshold

        return rmetrics

    else:
        raise UserWarning(f"eval for data type {cfg.data.type} not implemented")
