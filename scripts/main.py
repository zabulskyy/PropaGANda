letters = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя-\'"

lemma_dict = dict()

LEMMA_DICT_PATH = "../data/lemma_dict.txt"

with open("lemma_dict.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        l = line.split()
        lemma_dict[l[0]] = l[1]
        
        
def get_ascii(word, l=letters):
    new = ""
    for w in word:
        if w in l:
            new += w
    return new

def get_lemma_word(word, ld=lemma_dict):
    new_word = get_ascii(word.lower().strip())
    if new_word and new_word in ld:
        return ld[new_word]
    return ""

def get_lemma_sent(sent, ld=lemma_dict):
    new = []
    for word in sent.split():
        word = get_lemma_word(word).strip()
        if word: 
            new.append(word)
    return new

def get_lemma_par(par):
    new = []
    for sent in par.split('.'):
        sent = get_lemma_sent(sent).strip()
        if sent: 
            new.append(sent)
    return new
