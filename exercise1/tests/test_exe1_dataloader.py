import json
import pytest

from math import isclose
from pathlib import Path
import torch
import h5py
from tests import *


@pytest.fixture(scope="module")
def json_data():
    test_json = 'exe1_test.json'
    relative_path = Path(__file__).parent

    with Path(relative_path, test_json).open('r') as json_file:
        json_data = json.load(json_file)

    return json_data


@pytest.fixture(scope="module")
def cases_test(json_data):
    cases = [json_data[i] for i in json_data]
    return cases


@pytest.fixture(scope="module")
def seq_fasta(cases_test):
    relative_path = Path(__file__).parent
    return [relative_path.joinpath(case.get('seq_fasta', None)) for case in cases_test]


@pytest.fixture(scope="module")
def lbls(cases_test):
    relative_path = Path(__file__).parent

    labels_cases = []
    for case in cases_test:
        with Path(relative_path, case.get('lbls', None)).open('r') as f:
            labels_cases.append(json.load(f))

    return labels_cases


@pytest.fixture(scope="module")
def seqs(cases_test):
    return [case.get('seqs', None) for case in cases_test]


@pytest.fixture(scope="module")
def num_seqs(cases_test):
    return [case.get('num_seqs', None) for case in cases_test]


@pytest.fixture(scope="module")
def max_res(cases_test):
    return [case.get('max_res', None) for case in cases_test]


@pytest.fixture(scope="module")
def token_data(cases_test):
    relative_path = Path(__file__).parent

    h5_cases = []
    for case in cases_test:
        seq_tokens = {}
        with h5py.File(relative_path.joinpath(case.get('tokens', None)), 'r') as hf:
            for key in hf.keys():
                seq_tokens[key] = torch.from_numpy(hf[key][0][:, :]), torch.from_numpy(hf[key][1][:, :])
        h5_cases.append(seq_tokens)

    return h5_cases


@pytest.fixture(scope="module")
def embedding_data(cases_test):
    relative_path = Path(__file__).parent

    h5_cases = []
    for case in cases_test:
        embeddings = {}
        with h5py.File(relative_path.joinpath(case.get('embeddings', None)), 'r') as hf:
            for key in hf.keys():
                embeddings[key] = torch.from_numpy(hf[key][:])
        h5_cases.append(embeddings)

    return h5_cases


@pytest.fixture(scope="module")
def collate(cases_test):
    return [case.get('collate', None) for case in cases_test]


@pytest.fixture(scope="module")
def dataloader_data(cases_test):
    relative_path = Path(__file__).parent

    h5_cases = []
    for case in cases_test:
        with h5py.File(relative_path.joinpath(case.get('data', None)), 'r') as hf:
            h5_cases.append({run: [(torch.from_numpy(data['emb'][:]), torch.from_numpy(data['lbl'][:]))
                                   for data_idx, data in group.items()]
                             for run, group in hf.items()})

    return h5_cases


@pytest.fixture(scope="module")
def getitem_data(cases_test):
    relative_path = Path(__file__).parent

    h5_cases = []
    for case in cases_test:
        res_case = case.get('get_item', None)
        with h5py.File(relative_path.joinpath(res_case[1]), 'r') as hf:
            h5_cases.append((1, (torch.from_numpy(hf['emb'][:]), hf['lbl'][()])))

    return h5_cases


@pytest.fixture(scope="module")
def test_collate(collate):
    for coll in collate:
        input_collate = [(torch.Tensor(ten), i) for ten, i in coll['input']]
        try:
            from exe1_dataloader import collate_paired_sequences
            student_collate = collate_paired_sequences(input_collate)
        except Exception:
            raise AssertionError('Error in exe1_dataloader.collate_paired_sequences().') from None

        try:
            output = (torch.Tensor(coll['output'][0]), torch.Tensor(coll['output'][1]))
            passed = isinstance(student_collate, tuple) and (len(student_collate) == 2)
            assert passed, 'Return type should be a tuple of length 2!'
            passed = isinstance(student_collate[0], torch.Tensor) and isinstance(student_collate[1], torch.Tensor)
            assert passed, 'Return type should be tuple of two tensors!'
            passed = torch.allclose(student_collate[0], output[0], atol=1e-4) \
                     and torch.allclose(student_collate[1], output[1], atol=1e-4)
            assert passed, 'Collate function incorrect!'
        except AssertionError as msg:
            return False, msg

    return True, ''


def test_collate_x1(test_collate):
    passed, assertion_msg, *_ = test_collate
    assert passed, f'Failed test_collate(). {assertion_msg}'


def test_collate_x2(test_collate):
    passed, assertion_msg, *_ = test_collate
    assert passed, f'Failed test_collate(). {assertion_msg}'


@pytest.fixture(scope="module")
def token_seq_dataset(seq_fasta, lbls, max_res):
    try:
        from exe1_dataloader import TokenSeqDataset
        datasets = [TokenSeqDataset(seqs, lbl, res, "/workspace/protbert_weights", torch.device('cpu')) for seqs, lbl, res in zip(seq_fasta, lbls, max_res)]
        return datasets
    except Exception as e:
        raise AssertionError('Error in exe1_dataloader.TokenSeqDataset!') from None


@pytest.fixture(scope="module")
def test_model_loading(token_seq_dataset):
    try:
        from transformers import BertModel, BertTokenizer
        for dataset in token_seq_dataset:
            try:
                passed = isinstance(dataset.protbert, BertModel)
                assert passed, 'Model is not a BertModel!'
                passed = isinstance(dataset.tokenizer, BertTokenizer)
                assert passed, 'Tokenizer is not a BertTokenizer!'
            except AssertionError as msg:
                return False, msg
    except Exception:
        raise AssertionError('Error in exe1_dataloader.TokenSeqDataset!') from None

    return True, ''


def test_model_loading_x1(test_model_loading):
    passed, assertion_msg, *_ = test_model_loading
    assert passed, f'Failed test_model_loading(). {assertion_msg}'


def test_model_loading_x2(test_model_loading):
    passed, assertion_msg, *_ = test_model_loading
    assert passed, f'Failed test_model_loading(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_parse_fasta(token_seq_dataset, seq_fasta, seqs, num_seqs):
    for dataset, fasta, seq_dict, n_seq in zip(token_seq_dataset, seq_fasta, seqs, num_seqs):
        try:
            student_num_seqs = len(dataset)
            student_fasta = dataset.parse_fasta_input(fasta)
        except Exception:
            raise AssertionError('Error in exe1_dataloader.TokenSeqDataset.parse_fasta_input() or '
                                 'exe1_dataloader.TokenSeqDataset.__len__().') from None

        try:
            passed = (student_num_seqs == n_seq)
            assert passed, 'Number of parsed sequences incorrect! Check __len__().'
            passed = isinstance(student_fasta, dict)
            assert passed, 'Return type of parse_fasta_input() should be dict!'
            passed = (student_fasta == seq_dict)
            assert passed, 'Parsed sequences incorrect!'
        except AssertionError as msg:
            return False, msg

    return True, ''


def test_parse_fasta_x1(test_parse_fasta):
   passed, assertion_msg, *_ = test_parse_fasta
   assert passed, f'Failed test_parse_fasta(). {assertion_msg}'


def test_parse_fasta_x2(test_parse_fasta):
   passed, assertion_msg, *_ = test_parse_fasta
   assert passed, f'Failed test_parse_fasta(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_tokenize(token_seq_dataset, seqs, token_data):
    for dataset, seq_dict, sol_tokens in zip(token_seq_dataset, seqs, token_data):
        try:
            student_token_att = {}
            for idx, seq in seq_dict.items():
                student_token_att[idx] = dataset.tokenize(seq)
        except Exception:
            raise AssertionError('Error in exe1_dataloader.TokenSeqDataset.tokenize().') from None

        try:
            for idx, student_data in student_token_att.items():
                passed = isinstance(student_data, tuple) and (len(student_data) == 2)
                assert passed, 'Return type should be a tuple of length 2!'

                tok, att = student_data
                passed = isinstance(tok, torch.Tensor) and isinstance(att, torch.Tensor)
                assert passed, 'Return type should be tuple containing two Tensors!'
                passed = (tok.shape == sol_tokens[idx][0].shape)
                assert passed, 'Tokens have incorrect shape!'
                passed = (att.shape == sol_tokens[idx][1].shape)
                assert passed, 'Attention mask has incorrect shape!'

                passed = torch.all(sol_tokens[idx][0] == tok).item()
                assert passed, 'Tokens are incorrect!'
                passed = torch.all(sol_tokens[idx][1] == att).item()
                assert passed, 'Attention mask is incorrect!'
        except AssertionError as msg:
            return False, msg

    return True, ''


def test_tokenize_x1(test_tokenize):
   passed, assertion_msg, *_ = test_tokenize
   assert passed, f'Failed test_tokenize(). {assertion_msg}'


def test_tokenize_x2(test_tokenize):
   passed, assertion_msg, *_ = test_tokenize
   assert passed, f'Failed test_tokenize(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_embedding(token_seq_dataset, token_data, embedding_data):
    for dataset, tokens, sol_embeddings in zip(token_seq_dataset, token_data, embedding_data):
        try:
            student_embeddings = {}
            for idx, (tok, att) in tokens.items():
                student_embeddings[idx] = dataset.embedd(tok, att)

        except Exception as e:
            raise AssertionError('Error in exe1_dataloader.TokenSeqDataset.embedd().') from None

        try:
            for idx, student_data in student_embeddings.items():
                passed = isinstance(student_data, torch.Tensor)
                assert passed, 'Return type should be Tensor!'

                passed = (student_data.shape == sol_embeddings[idx].shape)
                assert passed, 'Embeddings have incorrect shape!'

                passed = torch.allclose(sol_embeddings[idx], student_data, atol=1e-4)
                assert passed, 'Embeddings are incorrect!'
        except AssertionError as msg:
            return False, msg

    return True, ''


def test_embedding_x1(test_embedding):
    passed, assertion_msg, *_ = test_embedding
    assert passed, f'Failed test_embedding(). {assertion_msg}'


def test_embedding_x2(test_embedding):
    passed, assertion_msg, *_ = test_embedding
    assert passed, f'Failed test_embedding(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_get_item(token_seq_dataset, getitem_data):
    for dataset, item in zip(token_seq_dataset, getitem_data):
        try:
            student_item = dataset[item[0]]
        except Exception:
            raise AssertionError('Error in exe1_dataloader.TokenSeqDataset.__getitem__().') from None

        try:
            passed = isinstance(student_item, tuple) and (len(student_item) == 2)
            assert passed, 'Return type should be a tuple of length 2!'

            emb, lbl = student_item
            passed = isinstance(emb, torch.Tensor) and isinstance(lbl, int)
            assert passed, 'Tuple should contain a Tensor and an integer!'

            passed = torch.allclose(emb, item[1][0], atol=1e-4)
            assert passed, 'Embedding incorrect!'
            passed = (lbl == item[1][1])
            assert passed, 'Label incorrect!'
        except AssertionError as msg:
            return False, msg

    return True, ''


def test_get_item_x1(test_get_item):
    passed, assertion_msg, *_ = test_get_item
    assert passed, f'Failed test_get_item(). {assertion_msg}'


def test_get_item_x2(test_get_item):
   passed, assertion_msg, *_ = test_get_item
   assert passed, f'Failed test_get_item(). {assertion_msg}'


@pytest.fixture(scope="module")
def test_dataloader(seq_fasta, lbls, dataloader_data, max_res):
    for seqs, lbl, reference_data, res in zip(seq_fasta, lbls, dataloader_data, max_res):
        try:
            from exe1_dataloader import get_dataloader
            student_dataloader = get_dataloader(seqs, lbl, 3, res, "/workspace/protbert_weights", torch.device('cpu'), 42)
        except Exception:
            raise AssertionError('Error in exe1_dataloader.get_dataloader().') from None

        try:
            passed = isinstance(student_dataloader, torch.utils.data.DataLoader)
            assert passed, 'Return type should be DataLoader!'
            passed = (len(student_dataloader) == len(reference_data))
            assert passed, 'Number of data points in the DataLoader incorrect!'

            for ref in reference_data.values():
                for idx, student_data in enumerate(student_dataloader):
                    passed = torch.allclose(student_data[0], ref[idx][0], atol=1e-4)
                    assert passed, 'Embeddings are incorrect! Is your dataloader deterministic w.r.t. the seed?'

                    passed = torch.allclose(student_data[1], ref[idx][1], atol=1e-4)
                    assert passed, 'Labels are incorrect! Is your dataloader deterministic w.r.t. the seed?'
        except AssertionError as msg:
            return False, msg

    return True, ''


def test_dataloader_x1(test_dataloader):
   passed, assertion_msg, *_ = test_dataloader
   assert passed, f'Failed test_dataloader(). {assertion_msg}'


def test_dataloader_x2(test_dataloader):
   passed, assertion_msg, *_ = test_dataloader
   assert passed, f'Failed test_dataloader(). {assertion_msg}'


def test_dataloader_x3(test_dataloader):
   passed, assertion_msg, *_ = test_dataloader
   assert passed, f'Failed test_dataloader(). {assertion_msg}'
