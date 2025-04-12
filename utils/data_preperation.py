from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jieba

def tokenize_text(text_data, tokenizer_type='basic_chinese'):
    
    if tokenizer_type == 'basic_chinese':
        # lcut 是jieba 的默认分词器，
        tokenizer = jieba.lcut
    elif tokenizer_type == 'basic_english':
        tokenizer = get_tokenizer('basic_english')
    # 如果 text_data 是一个字符串，则将其转换为列表
    if isinstance(text_data, str):
        return tokenizer(text_data)
    else:
        tokens = [tokenizer(line) for line in text_data]
    return tokens

def build_vocabulary(tokens, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
    # vocab_iterator 是一个生成器，用于迭代tokens中的每个token
    def vocab_iterator():
        for token in tokens:
            yield token
            
    vocab = build_vocab_from_iterator(vocab_iterator(), specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def build_vocabulary_with_data(tokens, tokenizer_type='basic_chinese',  specials=['<unk>', '<pad>', '<bos>', '<eos>']):
    
    # vocab_iterator 是一个生成器，用于迭代tokens中的每个token
    # 定义一个生成器函数，用于迭代所有文本数据的分词结果
    def vocab_iterator():
        for token in tokens:
            yield token
            
    vocab = build_vocab_from_iterator(vocab_iterator(), specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab