import re
import string

def preprocess_sentence(sentence):
    """
    Preprocess input sentence. 
    - Remove punctuation; 
    - Replace umlauts (e.g. ä = ae) 
    - Replace sharp s (ß = ss)
    - Encode into ascii and ignore errors; then decode again 

    Arguments
    sentence -- string

    Returns
    sentence -- preprocessed string

    """
    sentence = sentence.lower()
    sentence = re.sub("'", '', sentence)
    sentence = sentence.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')
    exclude = set(string.punctuation)
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    # enclose every sentence with "start_" and "_end"
    sentence = 'start_ ' + sentence + ' _end'
    sentence = sentence.encode("ascii", "ignore")
    sentence = sentence.decode()
    return sentence