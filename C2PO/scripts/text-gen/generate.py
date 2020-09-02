import os
import sys
import argparse
import torch
import random
import torch.nn.functional as F

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import src.models.utils as model_utils

prefixes = ['tries', 'starts', 'wants', 'begins', ]
random.seed(0)


def capitalizeFirst(phrase):
    words = phrase.split()
    words[0] = words[0].capitalize()
    return ' '.join(words)


def set_atomic_inputs(input_event, output_event, category, data_loader, text_encoder):
    in_prefix, _ = data.atomic_data.do_example(text_encoder, input_event, None, True, None)
    # out_prefix, _ = data.atomic_data.do_example(text_encoder, output_event, None, True, None)

    # XMB = torch.zeros(1, data_loader.max_event + 1 + len(out_prefix)).long().to(cfg.device)
    XMB = torch.zeros(1, data_loader.max_event + 1).long().to(cfg.device)

    XMB[:, :len(in_prefix)] = torch.LongTensor(in_prefix)
    XMB[:, data_loader.max_event] = torch.LongTensor([text_encoder.encoder["<{}>".format(category)]])
    # XMB[:, data_loader.max_event + 1:] = torch.LongTensor(out_prefix)

    batch = {}
    batch["sequences"] = XMB
    batch["attention_mask"] = data.atomic_data.make_attention_mask(XMB)

    return batch


def query(input_event, category='xNeed', sampling_algorithm='beam-10 '):
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    outputs = interactive.get_atomic_sequence(input_event, model, sampler, data_loader, text_encoder, category)[category]
    for k, v in outputs.items():
        outputs[k] = v[::-1]
    return outputs


def expand(event, width=3, depth=3, direction='xNeed'):
    if direction == 'forward':
        category = 'xWant'
    elif direction == 'backward':
        category = 'xNeed'

    stack = [[event]]
    done = set()

    outputs = []

    while len(stack) > 0:

        curr = stack.pop(0)
        if len(curr) != 1:
            outputs.append(curr)

        # don't repeat previous actions
        previous_action = curr[-1][len(character.split()) + 1:]
        if previous_action in done:
            continue
        done.add(previous_action)

        # get predictions
        out = query(curr[-1][len(character.split()) + 1:], category=category)

        reqs = out['beams'][:width]
        reqs = [[character, random.choice(prefixes)] + r.split() for r in reqs]

        # add new beams
        for r in reqs:
            if len(r) <= 3 or len(curr) >= depth or len(r) > 12:
                continue
            stack.insert(0, curr + [" ".join(r)])

    return outputs


def getProb(input_event, output_event, category='xNeed'):
    batch = set_atomic_inputs(input_event, output_event, category, data_loader, text_encoder)

    start_idx = data_loader.max_event + data.atomic_data.num_delimiter_tokens["category"]

    XMB = batch["sequences"][:, :start_idx]
    MMB = batch["attention_mask"][:, :start_idx]

    XMB = model_utils.prepare_position_embeddings(opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))

    beam_ll = 0
    for w in output_event.split():
        lm_probs = F.log_softmax(model(
            XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)
        dist = lm_probs[:, -1, :].squeeze()

        word = w + '</w>'
        # import ipdb; ipdb.set_trace()
        if word not in data_loader.vocab_encoder:
            return -1000
        else:
            tok_ll = dist[data_loader.vocab_encoder[w + '</w>']]
            next_tok = torch.tensor([[data_loader.vocab_encoder[w + '</w>']]], dtype=torch.long, device=MMB.device)

        beam_ll += tok_ll
        next_pos = XMB[:, -1:, 1] + 1

        next_x = torch.cat((next_tok, next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)
    return beam_ll


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_file", type=str,
                        default="/pretrained_models/atomic_pretrained_model.pickle")
    parser.add_argument('--file', type=str)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--outfile', type=str)
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    model.eval()

    file = open(args.file, 'r')
    output = open(args.outfile, 'w')
    with torch.no_grad():
        for line in file:
            parsed = line.split('\t')
            character = parsed[0]
            story = parsed[1:]
            out = []

            if args.device != "cpu":
                cfg.device = int(args.device)
                cfg.do_gpu = True
                torch.cuda.set_device(cfg.device)
                model.cuda(cfg.device)
            else:
                cfg.device = "cpu"

            for i in range(len(story) - 1):
                A = story[i]
                B = story[i + 1]

                endings = expand(B, direction='backward')
                beginnings = expand(A, direction='forward')

                all = []
                # print(endings,beginnings)
                for b in beginnings:
                    for e in endings:
                        b_action = ' '.join(b[-1].split()[len(character.split()) + 1:])
                        e_action = ' '.join(e[-1].split()[len(character.split()) + 1:])
                        forward_prob = getProb(b_action, e_action, category='xWant')
                        backward_prob = getProb(e_action, b_action, category='xNeed')

                        # normalize by probability of predicting "to"
                        forward_prob /= getProb(b[-1], 'to', category='xWant')
                        backward_prob /= getProb(e[-1], 'to', category='xNeed')

                        all.append((forward_prob + backward_prob, b + e[::-1]))

                all.sort()

                assert all[0][0] == min(all)[0]
                candidate = random.choice(all[:args.topk])[1]
                candidate = candidate[1:-1]
                print('. '.join(candidate))
                out.append('. '.join(candidate))

            for i in range(len(story) - 1):
                output.write(capitalizeFirst(story[i].strip()) + '. ' + capitalizeFirst(out[i]) + '. ')
            output.write(capitalizeFirst(story[-1].strip()) + '.')
            output.write('\n')
            output.flush()
