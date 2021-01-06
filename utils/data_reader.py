import torch
from torchtext import data

from .config import MIN_FREQ, MAX_LENGTH, MAX_VOCAB


def amazon_dataset_iters(
        parent_path, batch_sizes=(16, 64, 64),
        minimum_frequency=MIN_FREQ,
        verbose=True):
    """Helper function for creating batch iterators over Amazon Reviews datasets.

    Arguments:
        parent_path: path to folder with train.json, val.json and test.json in Amazon format
        device: device to allocate batches on
        batch_sizes: tuple of (train_batch_size, val_batch_size, test_batch_size)
        minimum_frequency: minimum frequency of words in batches (parameter of data.Field.build_vocab)
        verbose: True will print current status of processing

    Returns:
        (vocab, train_iter, val_iter, test_iter) - vocab and batch iterators for train, validation and test data.
        Each batch contains the following fields:
        batch.item - torch.LongTensor of shape (batch_size,) - numbers of reviewed items
        batch.user - torch.LongTensor of shape (batch_size,) - numbers of reviewers
        batch.text - torch.LongTensor of shape (word_num,batch_size) - encoded words in sentences
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Extract item and user data
    # Since we don't need tokenization, sequential is set to False
    item = data.Field(sequential=False)
    user = data.Field(sequential=False)

    # Extract review text data
    text = data.Field(
        sequential=True,
        tokenize='spacy',
        init_token="<sos>",
        eos_token="<eos>",
        fix_length=MAX_LENGTH + 2,
        lower=True)

    # Extract rating data
    # Since the rating is then fed into the embedding layer, it should also be dtype of torch.long(default)
    rating = data.Field(
        sequential=False,
        use_vocab=False)

    if verbose:
        print('Loading datasets...')

    train, val, test = data.TabularDataset.splits(
        path=parent_path,
        train='train.json',
        test='test.json',
        validation='val.json',
        format='json',
        fields={
            'asin': ('item', item),
            'reviewerID': ('user', user),
            'reviewText': ('text', text),
            'overall': ('rating', rating)
        }
    )
    if verbose:
        print('datasets loaded')
    item.build_vocab(train)
    if verbose:
        print('item vocab built')
    user.build_vocab(train)
    if verbose:
        print('user vocab built')
    # text.build_vocab(train.text, min_freq=minimum_frequency)
    text.build_vocab(train.text, min_freq=minimum_frequency, max_size=MAX_VOCAB)
    if verbose:
        print('text vocab built')

    train_iter, val_iter, test_iter = data.Iterator.splits(
        datasets=(train, val, test),
        batch_sizes=batch_sizes,
        repeat=False,
        sort=False,
        device=device
    )

    print('Number of batches in 1 training epoch: {}'.format(len(train_iter)))
    print('Number of batches in 1 validation epoch: {}'.format(len(val_iter)))
    print('Number of batches in 1 testing epoch: {}'.format(len(test_iter)))

    print('============== Dataset Loaded ==============')

    return item.vocab, user.vocab, text.vocab, train_iter, val_iter, test_iter
