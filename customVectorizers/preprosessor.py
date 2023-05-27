import re

def preprosessor_num(text):
    text = text.lower() 
    # simplifies all aritmetic expressions and nums to NUM
    text = re.sub(r'(\s|^)[\d\-\+\=\.\,\\]+\_NUM', ' NUM', text) 
    return text


def preprosessor_num_and_len(text):
    length = len(text.split(' ')) # num tokens in text
    processed = text.lower() 
    # simplifies all aritmetic expressions and nums to NUM
    processed = re.sub(r'(\s|^)[\d\-\+\=\.\,\\]+\_NUM', ' NUM', text)
    # hacked way of getting CountVectoriser to add feature for num tokens in document
    processed = text + " LEN"*length 
    return processed

def preprosessor_len(text):
    length = len(text.split(' ')) # num tokens in text
    # hacked way of getting CountVectoriser to add feature for num tokens in document
    processed = text + " LEN"*length 
    return processed

def remove_tags_1(text):
    # removes all words with ADJ ADV, PROPN, SYM and NUM word classes
    processed = re.sub(r'\b[\w\']*\_(ADJ|ADV|PROPN|SYM|NUM)\b', '', text)
    return processed

def remove_tags_2(text):
    # removes all words with ADV and PROPN, SYM and NUM word classes
    processed = re.sub(r'\b[\w\']*\_(ADV|PROPN|SYM|NUM)\b', '', text)
    return processed

def preprosessor_POS_count(text):
    n = text.count('NOUN')
    v = text.count('VERB')
    ad = text.count('ADJ')
    av = text.count('ADV')
    p = text.count('PROPN')
    s = text.count('SYM')
    num = text.count('NUM')
    length = len(text.split(' ')) # num tokens in text
    # hacked way of getting CountVectoriser to add feature for num tokens in document
    processed = text + " LEN"*length 
    processed = text + " NOUN"*n + " VERB"*v + " ADJ"*ad + " ADV"*av + " PROPN"*p + " SYM"*s + " NUM"*num
    return processed