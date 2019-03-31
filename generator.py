from funny_sentences import FunnySentences

CLF_PATH = 'algtest_classifyer.clf'
MCH_PATH = 'lm.bin'

def main():
	fs = FunnySentences(MCH_PATH, CLF_PATH)
	sent = fs.gen_funny(60) # , 'ибо Господь есть Бог.'
	print('\n\n'.join(sent))


main()