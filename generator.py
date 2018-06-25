import markovify
from funny_sentences import FunnySentences

CORPUS = 'data/corpus.txt'
BIBLE = 'data/bible.txt'
CODING = 'data/coding.txt'

def main():
	with open(CORPUS) as f:
		text = f.read()
	with open(BIBLE) as f:
		bible = f.read()
	with open(CODING) as f:
		coding = f.read()
	fs = FunnySentences(bible, coding)
	fs.learn()
	text_moldel = markovify.Text(text)
	sent = fs.gen_funny(text_moldel.make_sentence, 60)
	print('\n\n'.join(sent))


main()