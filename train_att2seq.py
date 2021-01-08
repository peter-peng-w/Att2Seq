import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
import math
import time
import torch.utils.tensorboard as tb
from os import path
from utils.data_reader import amazon_dataset_iters
from utils import config
from model.att2seq import Decoder, Encoder, Att2Seq

from nltk.translate import bleu_score
# from nltk.translate.bleu_score import sentence_bleu


# def test_review_bleu(gts_data, generate_data, vocab, bleu_totals, length, epoch):
#     type_wights = [
#         [1., 0, 0, 0],
#         [.5, .5, 0, 0],
#         [1 / 3, 1 / 3, 1 / 3, 0],
#         [.25, .25, .25, .25]
#     ]
#     write_file = './generate_sentence.txt'
#     sf = bleu_score.SmoothingFunction()

#     # batch first
#     gts_idx = torch.transpose(gts_data, 0, 1)
#     _, generate_idx = generate_data.max(2)
#     generate_idx = torch.transpose(generate_idx, 0, 1)

#     gts_sentence = []
#     gene_sentence = []
#     # detokenize the sentence
#     for token_ids in gts_idx:
#         current = [vocab.itos[id] for id in token_ids.detach().cpu().numpy()]
#         gts_sentence.append(current)
#     for token_ids in generate_idx:
#         current = [vocab.itos[id] for id in token_ids.detach().cpu().numpy()]
#         gene_sentence.append(current)

#     with open(write_file, 'at') as f:
#         for i in range(len(gts_sentence)):
#             print('Epoch: {0} || gt: {1} || gene: {2}'.format(epoch, gts_sentence[i], gene_sentence[i]), file=f)

#     # compute bleu score
#     assert len(gts_sentence) == len(gene_sentence)
#     for i in range(len(gts_sentence)):
#         refs = gts_sentence[i]
#         sample = gene_sentence[i]
#         if len(refs) == 0:
#             continue
#         length += 1
#         for j in range(4):
#             weights = type_wights[j]
#             bleu_totals[j] += bleu_score.sentence_bleu([refs], sample, smoothing_function=sf.method1, weights=weights)

#     return bleu_totals, length


def test_review_bleu_new(gts_data, generate_data, vocab, bleu_totals, length, epoch):
    type_wights = [
        [1., 0, 0, 0],
        [.5, .5, 0, 0],
        [1 / 3, 1 / 3, 1 / 3, 0],
        [.25, .25, .25, .25]
    ]
    write_file = './text_results/generate_sentence_new.txt'
    sf = bleu_score.SmoothingFunction()

    # batch first
    gts_idx = torch.transpose(gts_data, 0, 1)
    _, generate_idx = generate_data.max(2)
    generate_idx = torch.transpose(generate_idx, 0, 1)

    gts_sentence = []
    gene_sentence = []
    # detokenize the sentence
    for token_ids in gts_idx:
        current = []
        for id in token_ids.detach().cpu().numpy():
            if id == 1 or id == 2 or id == 3:
                pass
            else:
                current.append(vocab.itos[id])
            if id == 3:
                # 3 --- <eos>
                break
        gts_sentence.append(current)
    for token_ids in generate_idx:
        current = []
        for id in token_ids.detach().cpu().numpy():
            if id == 1 or id == 2 or id == 3:
                pass
            else:
                current.append(vocab.itos[id])
            if id == 3:
                # 3 --- <eos>
                break
        gene_sentence.append(current)

    with open(write_file, 'at') as f:
        for i in range(len(gts_sentence)):
            print('Epoch: {0} || gt: {1} || gene: {2}'.format(epoch, gts_sentence[i], gene_sentence[i]), file=f)

    # compute bleu score
    assert len(gts_sentence) == len(gene_sentence)
    for i in range(len(gts_sentence)):
        refs = gts_sentence[i]
        sample = gene_sentence[i]
        if len(refs) == 0:
            continue
        length += 1
        for j in range(4):
            weights = type_wights[j]
            bleu_totals[j] += bleu_score.sentence_bleu([refs], sample, smoothing_function=sf.method1, weights=weights)

    return bleu_totals, length


def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=1.0):
    model.train()
    epoch_loss = 0.0
    # running_loss = 0.0
    for i, batch in enumerate(iterator):
        user = batch.user
        # print('user shape: {}'.format(user.shape))
        item = batch.item
        # print('item shape: {}'.format(item.shape))
        rating = batch.rating
        # print('rating shape: {}'.format(rating.shape))
        text = batch.text
        # print('text shape: {}'.format(text.shape))
        # print('user: {}'.format(user))
        # print('item: {}'.format(item))
        # print('rating: {}'.format(rating))
        # print('text: {}'.format(text))
        optimizer.zero_grad()
        output = model(user, item, rating, text, teacher_forcing_ratio)        # output: (text_length, batch_size, output_dim(=text vocab size))
        output_dim = output.shape[-1]
        pred_output = output[1:].view(-1, output_dim)
        gt_text = text[1:].view(-1)
        # compute loss (cross entropy)
        loss = criterion(pred_output, gt_text)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # statistics
        epoch_loss += loss.item()
        # running_loss += loss.item()
    return epoch_loss / len(iterator)


def valid_epoch(model, iterator, criterion, epoch, text_vocab):
    model.eval()
    epoch_loss = 0.0
    bleu_totals = [0.] * 4
    num_reviews = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            user = batch.user
            item = batch.item
            rating = batch.rating
            text = batch.text
            output = model(user, item, rating, text, 0)
            output_dim = output.shape[-1]
            pred_output = output[1:].view(-1, output_dim)
            gt_text = text[1:].view(-1)
            loss = criterion(pred_output, gt_text)
            epoch_loss += loss.item()
            # compute bleu
            bleu_totals, num_reviews = test_review_bleu_new(text[1:], output, text_vocab, bleu_totals, num_reviews, epoch)
    bleu_totals = [bleu_total / num_reviews for bleu_total in bleu_totals]
    print('[%d] rating BLEU-1: %.3f' % (epoch + 1, bleu_totals[0]))
    print('[%d] rating BLEU-2: %.3f' % (epoch + 1, bleu_totals[1]))
    print('[%d] rating BLEU-3: %.3f' % (epoch + 1, bleu_totals[2]))
    print('[%d] rating BLEU-4: %.3f' % (epoch + 1, bleu_totals[3]))

    return epoch_loss / len(iterator)


def valid_epoch_without_bleu(model, iterator, criterion, epoch, text_vocab):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            user = batch.user
            item = batch.item
            rating = batch.rating
            text = batch.text
            output = model(user, item, rating, text, 0)
            output_dim = output.shape[-1]
            pred_output = output[1:].view(-1, output_dim)
            gt_text = text[1:].view(-1)
            loss = criterion(pred_output, gt_text)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def generate_review(model, device, user, item, rating, user_vocab, item_vocab, text_field, text_vocab, max_len=config.MAX_GENE_LEN):
    # user/item/rating: (batch_size(=1),)
    user = torch.LongTensor([user_vocab.stoi[user]]).to(device)
    item = torch.LongTensor([item_vocab.stoi[item]]).to(device)
    rating = torch.LongTensor([rating]).to(device)
    model.eval()
    # Feed data into the encoder
    with torch.no_grad():
        hidden, _, _, _ = model.encoder(user, item, rating)
        hidden = hidden.permute(2, 0, 1).contiguous()
    text_indexes = [text_vocab.stoi[text_field.init_token]]
    # Generation procedure
    for i in range(max_len):
        text_tensor = torch.LongTensor([text_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(text_tensor, hidden)
        pred_token = output.argmax(1).item()
        if pred_token == text_vocab.stoi[text_field.eos_token]:
            break
        text_indexes.append(pred_token)
    # Transform indexes to tokens in order to generate a review sentence
    text_tokens = [text_vocab.itos[i] for i in text_indexes]

    return text_tokens[1:]


def calculate_bleu(data, user_vocab, item_vocab, text_field, text_vocab, model, device, epoch, dataset='test', max_len=config.MAX_GENE_LEN):
    trgs = []
    pred_trgs = []

    type_weights = [
        [1., 0, 0, 0],
        [.5, .5, 0, 0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0],
        [.25, .25, .25, .25]
    ]

    write_file = './text_results/generate_{0}_{1}.txt'.format(dataset, epoch)
    sf = bleu_score.SmoothingFunction()
    bleu_totals = [0.] * 4
    cnt_bleu = 0

    for datum in data:
        user = vars(datum)['user']
        item = vars(datum)['item']
        rating = vars(datum)['rating']
        refs = vars(datum)['text']

        pred_text = generate_review(model, device, user, item, rating, user_vocab, item_vocab, text_field, text_vocab, max_len)

        trgs.append(refs)
        pred_trgs.append(pred_text)

        if len(refs) == 0:
            continue

        cnt_bleu += 1
        for i in range(4):
            weights = type_weights[i]
            bleu_totals[i] += bleu_score.sentence_bleu([refs], pred_text, smoothing_function=sf.method1, weights=weights)

    # write generated reviews into txt file
    with open(write_file, 'w') as f:
        for i in range(len(trgs)):
            print('GT: {0} || GE: {1}'.format(trgs[i], pred_trgs[i]), file=f)

    # compute BLEU score
    bleu_totals = [bleu_total / cnt_bleu for bleu_total in bleu_totals]
    return bleu_totals


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def init_weights_1(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def decay_lr(epoch, power=0.97):
    if epoch < 10:
        return 1.0
    else:
        return power ** (epoch - 9)


def train(args):
    # Load logger
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Loading the dataset
    dataset_folder = config.dataset_path
    item_vocab, user_vocab, text_vocab, text_field, train_iter, val_iter, test_iter, val_data, test_data = (
        amazon_dataset_iters(dataset_folder, batch_sizes=(config.train_batch, config.val_batch, config.test_batch))
    )

    # Count user and item number
    items_count = len(item_vocab)
    users_count = len(user_vocab)
    vocab_size = len(text_vocab)

    # Load model
    enc = Encoder(users_count, items_count)
    dec = Decoder(vocab_size, config.word_dim, config.enc_hid_dim, config.dec_hid_dim, config.rnn_layers, config.dropout)

    model = Att2Seq(enc, dec, device)

    model.to(device)

    model.apply(init_weights)

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=config.alpha)
    # TODO: Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lr)

    TEXT_PAD_IDX = text_vocab.stoi[text_field.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TEXT_PAD_IDX)

    global_step = 0
    best_valid_loss = float('inf')

    for epoch in range(args.num_epoch):
        start_time = time.time()
        # Training Procedure
        train_loss = train_epoch(model, train_iter, optimizer, criterion, config.CLIP, config.teacher_forcing)
        # Validation Procedure
        valid_loss = valid_epoch_without_bleu(model, val_iter, criterion, epoch, text_vocab)
        # valid_loss_train = valid_epoch(model, train_iter, criterion, epoch, text_vocab)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {current_lr}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss(valid): {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        # print(f'\t Val. Loss(train): {valid_loss_train:.3f} |  Val. PPL: {math.exp(valid_loss_train):7.3f}')

        # save model with best loss result on validation set
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './exp/att2seq_best.pth')
        # save model at some frequency
        if (epoch + 1) % args.save_model_freq == 0:
            torch.save(model.state_dict(), './exp/att2seq_{}_epoch.pth'.format(epoch+1))

        # Generating sentences on validation set and compute BLEU scores
        if (epoch + 1) % args.val_freq == 0:
            start_time = time.time()
            # Test on the val set
            bleu_scores = calculate_bleu(val_data, user_vocab, item_vocab, text_field, text_vocab, model, device, epoch=epoch+1, dataset='val')
            print('[VAL] [%d] rating BLEU-1: %.3f' % (epoch + 1, bleu_scores[0]*100))
            print('[VAL] [%d] rating BLEU-2: %.3f' % (epoch + 1, bleu_scores[1]*100))
            print('[VAL] [%d] rating BLEU-3: %.3f' % (epoch + 1, bleu_scores[2]*100))
            print('[VAL] [%d] rating BLEU-4: %.3f' % (epoch + 1, bleu_scores[3]*100))
            # write the bleu score to the logging file
            with open('./exp/logging.txt', 'at') as f:
                print('[VAL] [%d] rating BLEU-1: %.3f' % (epoch + 1, bleu_scores[0]*100), file=f)
                print('[VAL] [%d] rating BLEU-2: %.3f' % (epoch + 1, bleu_scores[1]*100), file=f)
                print('[VAL] [%d] rating BLEU-3: %.3f' % (epoch + 1, bleu_scores[2]*100), file=f)
                print('[VAL] [%d] rating BLEU-4: %.3f' % (epoch + 1, bleu_scores[3]*100), file=f)
            end_time = time.time()
            val_mins, val_secs = epoch_time(start_time, end_time)
            print(f'Val. Generate Time: {val_mins}m {val_secs}s')

        # Generating sentences on test set and compute BLEU scores
        if (epoch + 1) % args.test_freq == 0:
            start_time = time.time()
            # Test on the test set
            bleu_scores = calculate_bleu(test_data, user_vocab, item_vocab, text_field, text_vocab, model, device, epoch=epoch+1, dataset='test')
            print('[TEST] [%d] rating BLEU-1: %.3f' % (epoch + 1, bleu_scores[0]*100))
            print('[TEST] [%d] rating BLEU-2: %.3f' % (epoch + 1, bleu_scores[1]*100))
            print('[TEST] [%d] rating BLEU-3: %.3f' % (epoch + 1, bleu_scores[2]*100))
            print('[TEST] [%d] rating BLEU-4: %.3f' % (epoch + 1, bleu_scores[3]*100))
            # write the bleu score to the logging file
            with open('./exp/logging.txt', 'at') as f:
                print('[TEST] [%d] rating BLEU-1: %.3f' % (epoch + 1, bleu_scores[0]*100), file=f)
                print('[TEST] [%d] rating BLEU-2: %.3f' % (epoch + 1, bleu_scores[1]*100), file=f)
                print('[TEST] [%d] rating BLEU-3: %.3f' % (epoch + 1, bleu_scores[2]*100), file=f)
                print('[TEST] [%d] rating BLEU-4: %.3f' % (epoch + 1, bleu_scores[3]*100), file=f)
            end_time = time.time()
            test_mins, test_secs = epoch_time(start_time, end_time)
            print(f'Test Generate Time: {test_mins}m {test_secs}s')

    # Testing Procedure
    print('Finished training, start testing ...')
    model.load_state_dict(torch.load('./exp/att2seq_best.pth'))
    test_loss = valid_epoch(model, test_iter, criterion, args.num_epoch, text_vocab)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    # Test on the test set
    bleu_scores = calculate_bleu(test_data, user_vocab, item_vocab, text_field, text_vocab, model, device, epoch='final', dataset='test')
    print('[TEST] [final] rating BLEU-1: %.3f' % (bleu_scores[0]*100))
    print('[TEST] [final] rating BLEU-2: %.3f' % (bleu_scores[1]*100))
    print('[TEST] [final] rating BLEU-3: %.3f' % (bleu_scores[2]*100))
    print('[TEST] [final] rating BLEU-4: %.3f' % (bleu_scores[3]*100))
    # write the bleu score to the logging file
    with open('./exp/logging.txt', 'at') as f:
        print('[TEST] [final] rating BLEU-1: %.3f' % (bleu_scores[0]*100), file=f)
        print('[TEST] [final] rating BLEU-2: %.3f' % (bleu_scores[1]*100), file=f)
        print('[TEST] [final] rating BLEU-3: %.3f' % (bleu_scores[2]*100), file=f)
        print('[TEST] [final] rating BLEU-4: %.3f' % (bleu_scores[3]*100), file=f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./logging', help='The path of the logging dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-sf', '--save_model_freq', type=int, default=5, help='Frequency of saving model, per epoch')
    parser.add_argument('-s', '--save_dir', type=str, default='./exp', help='The path of experiment model dir')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for traning')
    parser.add_argument('-vf', '--val_freq', type=int, default=5, help='Frequency of testing (generate text) model on validset and compute scores, per epoch')
    parser.add_argument('-tf', '--test_freq', type=int, default=5, help='Frequency of testing (generate text) model on testset and compute scores, per epoch')
    args = parser.parse_args()
    train(args)
