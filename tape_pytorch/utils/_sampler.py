"""Implementation of a bucketed data sampler from PyTorch-NLP.
Modified by Roshan Rao.

See https://github.com/PetrochukM/PyTorch-NLP/
"""
import typing
import math
import operator
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self,
                 dataset,
                 sort_key: typing.Callable[[int], typing.Any],
                 indices: typing.Optional[typing.Iterable[int]] = None):
        super().__init__(dataset)
        self.dataset = dataset
        self.sort_key = sort_key
        if indices is None:
            sort_keys = map(sort_key, dataset)
        else:
            sort_keys = ((i, sort_key(dataset[i])) for i in indices)
        self.sorted_indices = [i for i, _ in sorted(sort_keys, key=operator.itemgetter(1))]

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.dataset)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.
    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted
    and vice versa. Provides ~10-25 percent speedup.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular
        libraries like ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together
        examples with a similar size length to reduce the padding required for each batch
        while maintaining some noise through bucketing.

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size
            would be less than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    Example:
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key,
                 dataset,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.dataset = dataset
        self.bucket_sampler = BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)), False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(self.dataset, self.sort_key, indices=bucket)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)
