# Copyright 2017 Mohammed H. Jabreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import

from tnet.dataset import Dataset
import gzip
from tnet.utils import get_file
from six.moves import cPickle
import sys
import numpy as np


class MNISTDataset(Dataset):
    """docstring for MNISTDataset."""
    def __init__(self, data):
        super(MNISTDataset, self).__init__()
        self.add_attribute("input", np.ndarray)
        self.add_attribute("target", np.ndarray)
        self._dataset = data

    def _get(self, idx):
        return self._dataset[0][idx], self._dataset[1][idx].astype(np.int32)

    @property
    def size(self):
        return self._dataset[0].shape[0]


def get_data(path='mnist.pkl.gz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.pkl.gz')

    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        [X_train, y_train], [X_test, y_test] = cPickle.load(f)
    else:
        [X_train, y_train], [X_test, y_test] = cPickle.load(f, encoding='bytes')

    f.close()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    train_set = MNISTDataset([X_train, y_train])
    #test_set = MNISTDataset([X_test, y_test])
    return train_set, [X_test, y_test]#test_set
