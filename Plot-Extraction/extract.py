from openie import StanfordOpenIE
import spacy
import neuralcoref
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

file = open(args.file)
text = file.read()
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

num_examples = 1

pronouns = ["all", "another", "any", "anybody", "anyone", "anything", "both", "each", "each", "other", "either", "everybody", "everyone", "everything", "few", "he", "her", "hers", "herself", "him", "himself", "his", "through", "it", "its",
            "itself", "little", "many", "me", "mine", "more", "most", "much", "my", "myself", "neither", "no", "one", "nobody", "none", "nothing", "one", "one", "another", "other", "others", "our", "ours", "ourselves", "through", "several",
            "she", "some", "somebody", "someone", "something", "that", "their", "theirs", "them", "themselves", "these", "they", "this", "those", "us", "we", "what", "whatever", "which", "whichever", "who", "whoever", "whom", "whomever",
            "whose", "you", "your", "yours", "yourself", "yourselves", ]

objective_pronouns = ["all", "another", "any", "anybody", "anyone", "anything", "both", "each", "each", "other", "either", "few", "her", "hers", "herself", "him", "himself", "his", "through", "its",
                      "itself", "little", "many", "me", "mine", "more", "most", "much", "my", "myself", "neither", "no", "nobody", "none", "nothing", "another", "other", "others", "our", "ours", "ourselves", "through", "several",
                      "some", "somebody", "someone", "something", "that", "their", "theirs", "them", "themselves", "these", "this", "those", "us", "what", "whatever", "which", "whichever", "who", "whoever", "whom", "whomever",
                      "whose", "your", "yours", "yourself", "yourselves", ]


def capitalizeFirst(phrase):
    words = phrase.split()
    words[0] = words[0].capitalize()
    return ' '.join(words)


def lowerFirst(phrase):
    words = phrase.split()
    words[0] = words[0].lower()
    return ' '.join(words)


def printMentions(doc):
    mentions = []
    for cluster in doc._.coref_clusters:
        mentions.append(cluster.mentions)
    return mentions


def intersect(b1, b2):
    return b1[1] > b2[0] and b1[0] < b2[1]


# extract coreference mentions and open ie triples
doc = nlp(text)
mentions = printMentions(doc)
with StanfordOpenIE() as client:
    core_nlp_output = client.annotate(text, simple_format=False)
    triples = []
    offset = 0
    for sentence in core_nlp_output['sentences']:
        for triple in sentence['openie']:

            # use character offset because openie and spacy disagree on word level offset
            for part in ['subject', 'relation', 'object']:
                span = part + 'Span'
                start = sentence['tokens'][triple[span][0]]['characterOffsetBegin']
                end = sentence['tokens'][triple[span][1]]['characterOffsetEnd']

                triple[span][0] = start
                triple[span][1] = end

            triples.append(triple)

characters = []
candidates = []
# find longest common in mentions to get list of characters
modifiers = ['the', 'her', 'its', 'his', 'their', 'a', 'this', 'that', 'those', 'these']
for m in mentions:
    best = ' '.join([w for w in m[0].text.split() if w not in modifiers])

    for c in m:

        if len(best) == 0:
            best = ' '.join([w for w in c.text.split() if w not in modifiers])

        if c.text.lower() in pronouns:
            continue
        str1 = best
        str2 = ' '.join([w for w in c.text.split() if w not in modifiers])

        match = SequenceMatcher(None, str1.split(), str2.split()).find_longest_match(0, len(str1.split()), 0, len(str2.split()))
        if match.size > 0:
            best = " ".join(str1.split()[match.a: match.a + match.size])
    if best in pronouns:
        continue

    # somtimes extracts something with alot of rephrasing so take the first
    best = best.split(',')[0]
    characters.append(best)
    candidates.append(m)

good = []
for char, cand in zip(characters, candidates):

    if args.verbose:
        print('=======================')
        print(char + '\n')
    used = []
    relationships = set()
    seq = [char]

    # map mentions to triples
    for mention in cand:

        # use char level
        m_range = [mention.start_char, mention.end_char]
        for t in triples:

            candidate = " ".join([capitalizeFirst(t['subject']), lowerFirst(t['relation'].replace('_', ' ')), t['object']])

            # if relation already used, then ignore
            bad = False
            for u in used:
                if intersect(t['relationSpan'], u) or t['relation'] in relationships:
                    bad = True
            if bad:
                continue

            if intersect(t['subjectSpan'], m_range):  # or intersect(t['objectSpan'], m_range):
                seq.append(candidate)
                used.append(t['relationSpan'])
                relationships.add(t['relation'])
                break
        if args.verbose:
            print(candidate)

    if len(seq) >= 5:
        good.append(seq)

random.seed(1)
f = open(args.outfile, 'a')
random.shuffle(good)
for i in range(num_examples):
    seq = good[i]
    for i in range(3):
        f.write('\t'.join([s for s in seq]))
        f.write('\n')
    print(seq[0], len(seq))
