if __name__ == "__main__":
    lemma_dict = get_lemma_dict()

def get_lemma_dict(path="data/lemma_dict.txt"):
    lemma_dict = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split()
            lemma_dict[l[0]] = l[1]
    return lemma_dict


def get_ascii(word):
    l = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя-\'"
    s = "!?.;\"'/\\,;()"
    new = ""
    for w in word:
        if w in l:
            new += w
        elif w in s and (new and new[-1] != ' '):
            new += " "
    return new

def get_lemma_word(word, ld):
    new_word = get_ascii(word.lower().strip())
    words = [x.strip() for x in new_word.split()]
    if len(words) <= 1:
        if new_word and new_word in ld:
            return [ld[new_word]]
    else:
        res = []
        for word in words:
            if word and word in ld:
                res.append(ld[word])
        return res
    return [""]

def get_lemma_sent(sent, ld):
    new = []
    for word in sent.split():
        word = get_lemma_word(word, ld)
        if word and word != [""]: 
            for w in word:
                new.append(w)
    return new

def get_lemma_par(par, ld):
    new = []
    for sent in par.split('.'):
        sent = get_lemma_sent(sent, ld)
        if sent: 
            new.append(sent)
    return new
