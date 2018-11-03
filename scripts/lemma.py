letters = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя-\'"

def get_lemma_dict(path="data/lemma_dict.txt"):
    lemma_dict = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split()
            lemma_dict[l[0]] = l[1]
    return lemma_dict

if __name__ == "__main__":
    lemma_dict = get_lemma_dict()


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
        sent = get_lemma_sent(sent)
        if sent: 
            new.append(sent)
    return new
