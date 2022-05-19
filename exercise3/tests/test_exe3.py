import json
import pytest

from math import isclose
from pathlib import Path
import torch
import h5py
from tests import *


@pytest.fixture(scope="module")
def json_data():
    test_json = 'exe3_test.json'
    relative_path = Path(__file__).parent

    with Path(relative_path, test_json).open('r') as json_file:
        json_data = json.load(json_file)

    return json_data


@pytest.fixture(scope="module")
def val_dataloader(json_data):
    relative_path = Path(__file__).parent

    with h5py.File(relative_path.joinpath(json_data.get('val_dataloader', None)), 'r') as hf:
        data = [(torch.from_numpy(data['emb'][:]), torch.from_numpy(data['lbl'][:]))
                for data_idx, data in hf['first'].items()]

    return data


@pytest.fixture(scope="module")
def val_model_path(json_data):
    return json_data.get('model_path', None)


@pytest.fixture(scope="module")
def test_model_loading(val_model_path, val_dataloader):
    try:
        from exe3_network import Model
        relative_path = Path(__file__).parent.parent

        checkpoint = torch.load(relative_path.joinpath('hypopt_model.pth'))
        model = Model(**checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
    except Exception:
        raise AssertionError('Error while loading the model.') from None


    try:
        from torch.nn import BCELoss
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data

                outputs = model.predict(inputs)

                loss = BCELoss()(outputs.float(), labels.float())
    except Exception:
        raise AssertionError('Error during forward pass.') from None

    return True, ''


def test_model(test_model_loading):
    passed, assertion_msg, *_ = test_model_loading
    assert passed, f'Failed test_model_loading(). {assertion_msg}'
