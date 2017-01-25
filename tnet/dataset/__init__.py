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
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from tnet.base import EventArgs, EventHook

__all__ = [
    "Dataset",
    "BatchDataset",
    "BatchingPolicies",
    "ResampleDataset",
    "ShuffleDataset",
    "DatasetIterator"
]


class Attribute(object):

    def __init__(self, dataset, name, dtype='float32'):

        self._name = name
        self._dtype = dtype
        self._owner = dataset
        self._index = None


    def __hash__(self):
         return hash((self._name, self._dtype, self._index))

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def owner(self):
        return self._owner

class Dataset(object):

    def __init__(self, attributes=[]):
        self._attributes = attributes
        self._data = None


    @property
    def size(self):
        raise NotImplementedError

    def _get(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self._get(idx)

    def __len__(self):
        return self.size

    def add_attribute(self, name, dtype=np.float32):

        if isinstance(name, Attribute):
            a = name
        else:
            assert type(name) == str
            a = Attribute(self, name, dtype)

        a._index = len(self._attributes)
        self._attributes.append(a)

    def get_attribute(self, key):

        if type(key) == str:
            for a in self._attributes:
                if a.name == key:
                    return a
        elif type(key) == int:
            for a in self._attributes:
                if a._index == key:
                    return a

        return None

    @property
    def attributes(self):
        return self._attributes



    def add_sample(self, values):
        sample = []
        for idx, v in enumerate(values):
            a = self.get_attribute(idx)
            assert type(v) == a.dtype
            sample.append(v)

        self._data.append(sample)



    def get_values(self, attribute):

        if isinstance(attribute, Attribute):
            idx = attribute._index
        else:
            assert type(attribute) == int

            idx = attribute



class BatchingPolicies:
    INCLUDE_LAST, SKIP_LAST, DIVISIBLE_ONLY = range(3)

class BatchDataset(Dataset):

    def __init__(self, dataset, batch_size, policy=BatchingPolicies.INCLUDE_LAST):

        assert isinstance(dataset, Dataset)
        assert type(batch_size) == int

        self._dataset = dataset
        self._batch_size = batch_size
        self._policy = policy

        if policy == BatchingPolicies.INCLUDE_LAST:

            self._size = int(math.ceil(dataset.size / batch_size))

        elif policy == BatchingPolicies.SKIP_LAST:

            self._size = int(math.floor(dataset.size / batch_size))

        elif policy == BatchingPolicies.DIVISIBLE_ONLY:

            assert dataset.size % batch_size == 0
            self._size = int(dataset.size / batch_size)

        else:
            raise Exception("Wrong batching policy is passed")



    def _get(self, idx):

        assert idx >= 0 and idx < self._size

        samples = dict([(a, []) for a in self._dataset.attributes])

        maxidx = self._dataset.size

        for i in range(self._batch_size):

            idx_ = idx * self._batch_size + i

            if idx_ >= maxidx:
                break

            s = self._dataset[idx_]
            for j, a in enumerate(self._dataset.attributes):
                samples[a].append(s[j])

        samples = dict([(a.name, np.array(samples[a])) for a in self._dataset.attributes])

        return samples


    @property
    def size(self):
        return self._size


class ResampleDataset(Dataset):

    def __init__(self, dataset, size=None):
        assert isinstance(dataset, Dataset)
        super(ResampleDataset, self).__init__()

        self._dataset = dataset
        if size is None:
            size = dataset.size

        self._sampling_size = size


    @property
    def size(self):
        return self._sampling_size

    def _get(self, idx):
        assert idx >= 0 and idx < self.size
        idx = self._sampler(idx)
        assert idx >= 0 and idx < self.size
        return self._dataset[idx]

    def _sampler(self, idx):
        raise NotImplementedError


class ShuffleDataset(ResampleDataset):

    def __init__(self, dataset):
        super(ShuffleDataset, self).__init__(dataset)
        self.resample()

    def _sampler(self, idx):
        return self.__perm[idx]

    def resample(self):
        self.__perm = np.random.permutation(self.size)



class OnSampleEventArgs(EventArgs):

    def __init__(self, sample):
        super(OnSampleEventArgs, self).__init__()
        self.sample = sample


class DatasetIterator(object):

    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self._dataset = dataset
        self.on_sample = EventHook()

    def _perm(self, idx):
        return idx

    def _filter(self, sample):
        return True

    def _transform(self, sample):
        return sample

    def _run(self):

        size = self._dataset.size
        
        idx = 0
        while idx < size:
            pidx = self._perm(idx)
            sample = self._transform(self._dataset[pidx])
            idx += 1
            if self._filter(sample):

                self.on_sample.invoke(OnSampleEventArgs(sample))

                yield sample

    def __call__(self):
        return self._run()
