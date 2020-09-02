from pytorch_transformers import BertTokenizer, BertForMaskedLM
import torch
import random
import numpy as np
from pytorch_transformers import BertTokenizer, BertForMaskedLM
import nltk
import argparse
import string

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='../PlotExtraction/fairy.txt')
parser.add_argument('--outfile', type=str, default='bert_fairy.txt')
parser.add_argument('--model', type=str, default='./bert/fairy')

args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.model)
model = BertForMaskedLM.from_pretrained(args.model, output_attentions=False)
model.eval()


def capitalizeFirst(phrase):
    words = phrase.split()
    words[0] = words[0].capitalize()
    return ' '.join(words)


def is_punctuation(s):
    return len(set(s).intersection(set(string.punctuation))) > 0


def getScore(sentence):
    tokenized_text = tokenizer.tokenize('[CLS] ' + "[MASK] " + sentence + ' [SEP]')
    mask_idxs = duplicates(tokenized_text, "[MASK]")

    if decoding_type == 'right to left':
        focus_mask_idx = max(mask_idxs)
    else:
        focus_mask_idx = random.choice(mask_idxs)

    mask_idxs.pop(mask_idxs.index(focus_mask_idx))
    temp_tokenized_text = tokenized_text.copy()
    temp_tokenized_text = [j for i, j in enumerate(temp_tokenized_text) if i not in mask_idxs]
    temp_indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
    ff = [idx for idx, i in enumerate(temp_indexed_tokens) if i == 103]
    temp_segments_ids = [0] * len(temp_tokenized_text)
    tokens_tensor = torch.tensor([temp_indexed_tokens])
    segments_tensors = torch.tensor([temp_segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    # get score of punctuation and compare to predicted score
    end_score = predictions[0, ff][0, tokenizer.convert_tokens_to_ids('.')]
    return end_score


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


file = open(args.file, 'r')
output = open(args.outfile, 'w')
for line in file:
    parsed = line.split('\t')
    character = parsed[0]
    story = parsed[1:]

    out = []
    for i in range(len(story) - 1):
        prev = story[i] + '.'
        next = story[i + 1] + '.'
        sentences = []
        for sentence_count in range(3):
            length = random.randint(3, 5)
            generated = '.'
            for i in range(length):
                decoding_type = 'right to left'

                tmp = character + ' ' + "[MASK] " * (length - i) + generated
                fill = ' '.join(['[CLS]', prev, tmp, next, '[SEP]'])
                # print(fill)
                tokenized_text = tokenizer.tokenize(fill)
                mask_idxs = duplicates(tokenized_text, "[MASK]")

                if decoding_type == 'right to left':
                    focus_mask_idx = max(mask_idxs)
                else:
                    focus_mask_idx = random.choice(mask_idxs)

                mask_idxs.pop(mask_idxs.index(focus_mask_idx))
                temp_tokenized_text = tokenized_text.copy()
                temp_tokenized_text = [j for i, j in enumerate(temp_tokenized_text) if i not in mask_idxs]
                temp_indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
                ff = [idx for idx, i in enumerate(temp_indexed_tokens) if i == 103]
                temp_segments_ids = [0] * len(temp_tokenized_text)
                tokens_tensor = torch.tensor([temp_indexed_tokens])
                segments_tensors = torch.tensor([temp_segments_ids])

                with torch.no_grad():
                    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                    predictions = outputs[0]

                k = 20

                predicted_token = '.'
                while is_punctuation(predicted_token):
                    predicted_index = random.choice(predictions[0, ff].argsort()[0][-k:]).item()
                    predicted_score = predictions[0, ff][0, predicted_index]
                    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                tokenized_text[focus_mask_idx] = predicted_token

                # get score of punctuation and compare to predicted score
                # end_score = getScore(character + ' ' + predicted_token + ' ' + generated + ' ' + sentence)
                # print(end_score)
                # if end_score > 5:
                #     break
                if generated != '.':
                    generated = ' ' + generated
                generated = predicted_token + generated

            final = capitalizeFirst(character) + ' ' + generated
            next = final + ' ' + next
            print(final)
            sentences.append(final)
        out.append(' '.join(sentences))

    print(len(out), len(story))
    for i in range(len(story) - 1):
        output.write(story[i] + '. ' + out[i] + ' ')
    output.write(story[-1] + '.')
    output.write('\n')
