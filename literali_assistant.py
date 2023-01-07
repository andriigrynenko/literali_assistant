import cv2
import os
import urllib

import webcam_utils
import char_net

characters = char_net.characters

def read_dict():
    dict_cache_path = './dict_cache.lst'
    dict_url = 'https://github.com/brown-uk/dict_uk/blob/master/data/dict/base.lst?raw=true'
    
    if not os.path.exists(dict_cache_path):
        urllib.request.urlretrieve(dict_url, dict_cache_path)

    file = open(dict_cache_path, encoding="utf8")
    dict = []
    for line in file.readlines():
        try:
            characters.index(line[0])
            word = line.split(' ')[0]
            for c in word:
                characters.index(c)
            if len(word) < 3:
                continue 
            dict.append(word)
        except:
            pass

    return dict

def gen_combo_words(words, intersection):
    suffix_dict = {}
    for w in words:
        if len(w) <= intersection:
            continue
        suffix_dict[w[-intersection:]] = w

    res = []

    for w in words:
        if len(w) <= intersection:
            continue
        if w[:intersection] in suffix_dict:
            res.append(suffix_dict[w[:intersection]] + w[intersection:])

    return res

dict = read_dict()
combo_words = []
for intersection in range(2, 5):
    combo_words += gen_combo_words(dict, intersection)
dict += combo_words
dict = set(dict)

def with_mask(dict):
    res = []
    for word in dict:
        mask = 0
        for i, c in enumerate(characters):
            try:
                word.index(c)
                mask |= 1<<i
            except:
                pass
        res.append((word, mask))
    return res

dict_with_mask = with_mask(dict)

def score_len(word):
    return len(word)

def score_len8(word):
    if len(word) < 8:
        return 0
    return 100 - len(word)

def find_match(chars, score = score_len):
    s = -1
    r = ""
    d = {}
    mask = 0
    for c in characters:
        d[c] = 0
    for i, c in enumerate(chars):
        if c == '?':
            continue
        d[c] += 1
        mask |= 1 << i

    for word, word_mask in dict_with_mask:
        if (mask & word_mask) != word_mask:
            continue
        ok = True
        for c in word:
            d[c] -= 1
            if d[c] < 0:
                ok = False
        for c in word:
            d[c] += 1
        
        if not ok:
            continue

        ns = score(word)
        if ns > s:
            s = ns
            r = word
    return r

print('Dict size:', len(dict))

net = char_net.CharNet.create_from_file('./char_net0.pt')

def process(extracted_squares):
    result = []
    for square in extracted_squares:
        character = net.guess_character(square)
        if not character is None:
            result.append(character)
        else:
            result.append('?')
    match = find_match(result)
    highlight_contour_ids = []
    if len(match) > 0:
        print(match)
        for c in match:
            id = result.index(c)
            highlight_contour_ids.append(id)
            result[id] = '*'
    return highlight_contour_ids

webcam_utils.main_loop(process)
