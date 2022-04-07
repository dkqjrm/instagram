import os
import json
import torch
import emoji
import re
import operator
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class HashtagDataset(Dataset):
    def __init__(self, img, text, hashtags, hashtags_onehot):
        self.img = img
        self.text = text
        self.hashtags = hashtags
        self.hashtags_onehot = hashtags_onehot

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return {"img": self.img[idx], "text": self.text[idx], "hashtags": self.hashtags[idx],
                "hashtags_onehot": self.hashtags_onehot[idx]}


def hashcollator(samples):
    img = []
    text = []
    hashtags = []
    hashtags_onehot = []
    text_maxlen = 0
    for sample in samples:
        img.append(sample['img'])
        text.append(sample['text'])
        if len(sample['text']) > text_maxlen:
            text_maxlen = len(sample['text'])
        hashtags.append(sample['hashtags'])
        hashtags_onehot.append(sample['hashtags_onehot'])
    for i in range(len(text)):
        text[i].extend([0 for q in range(text_maxlen - len(text[i]))])

    return torch.stack(img), torch.tensor(text), torch.LongTensor(hashtags), torch.LongTensor(hashtags_onehot)


class PreprocessDataset:
    def __init__(self):
        # 개별 post
        self.img = []
        self.text = []
        self.hashtags = []

        # vocab 정보들
        self.hashtag_vocab = {}  # {"hashtag1":0, "hashtag2":1, ...}
        self.hashtag_freq = {}  # {"hashtag" : count, ...}
        self.word_vocab = {"<PAD>": 0}  # {"word1":0, "word2":1, ...}
        self.word_freq = {"<PAD>": 0}  # {"word" : count}
        self.hashtag_vocab_size = 0
        self.word_vocab_size = 1

    def add_data(self, img, text, hashtags):
        self.img.append(img)
        self.text.append(text)
        self.hashtags.append(hashtags)
        for hashtag in hashtags:
            if hashtag not in self.hashtag_freq.keys():
                self.hashtag_freq[hashtag] = 1
                self.hashtag_vocab[hashtag] = self.hashtag_vocab_size
                self.hashtag_vocab_size += 1
            else:
                self.hashtag_freq[hashtag] += 1

        for word in text.split():
            if word not in self.word_freq.keys():
                self.word_freq[word] = 1
                self.word_vocab[word] = self.word_vocab_size
                self.word_vocab_size += 1
            else:
                self.word_freq[word] += 1

    def remove_low_freq(self):
        '''
        해시태그는 5645개 중 상위 2000개, 단어는 3147개 중 상위 3000개만 남김
        '''
        # print("vocab_size:", self.word_vocab_size)
        # print("hashtag_size:", self.hashtag_vocab_size)
        # print("hashtags:", self.hashtag_freq)
        # sorted_hashtag = sorted(self.hashtag_freq.values(), reverse=True)
        # sorted_vocab = sorted(self.word_freq.values(), reverse=True)
        # plt.plot(sorted_hashtag)
        # plt.show()
        # plt.plot(sorted_vocab)
        # plt.show()

        new_hashtag_vocab = {}  # {"hashtag1":0, "hashtag2":1, ...}
        sorted_tags = sorted(self.hashtag_freq.items(), key=operator.itemgetter(1), reverse=True)
        for i, each_tuple in enumerate(sorted_tags[:1000]):
            new_hashtag_vocab[each_tuple[0]] = i

        new_word_vocab = {}  # {"word1":0, "word2":1, ...}
        sorted_words = sorted(self.word_freq.items(), key=operator.itemgetter(1), reverse=True)
        for i, each_tuple in enumerate(sorted_words[:3000]):
            new_word_vocab[each_tuple[0]] = i

        new_img = []
        new_text = []
        new_hashtags = []
        new_hashtags_onehot = []
        for i in tqdm(range(len(self.text))):
            new_tags = []
            new_tags_onehot = [0 for q in range(len(new_hashtag_vocab))]
            for tag in self.hashtags[i]:
                if tag in new_hashtag_vocab.keys():
                    new_tags.append(new_hashtag_vocab[tag])
                    new_tags_onehot[new_hashtag_vocab[tag]] = 1
            new_words = []
            for word in self.text[i].split():
                if word in new_word_vocab.keys():
                    new_words.append(new_word_vocab[word])
            if new_words and new_tags:
                for each_tag in new_tags:
                    new_img.append(self.img[i])
                    new_text.append(new_words)
                    new_hashtags.append(each_tag)
                    new_hashtags_onehot.append(new_tags_onehot)

        self.img = new_img
        self.text = new_text
        self.hashtags = new_hashtags
        self.hashtag_vocab = new_hashtag_vocab
        self.hashtag_vocab_size = len(new_hashtag_vocab)
        self.word_vocab = new_word_vocab
        self.word_vocab_size = len(new_word_vocab)
        self.new_hashtags_onehot = new_hashtags_onehot


def word_process(word, use_emoji):  # emoji 처리
    if use_emoji:
        return emoji.demojize(word).replace(':', ' ') + ' '
    else:
        return word.encode('ascii', 'ignore').decode('ascii') + ' '


def remove_space(txt):
    txt = txt.replace('  ', ' ')
    if '  ' in txt:
        remove_space(txt)
    else:
        return txt


def create_raw_data(path, use_emoji):
    hashtags = os.listdir(path)
    hashdataset = PreprocessDataset()
    img_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    backbone.eval()
    if torch.cuda.is_available():
        backbone.to('cuda')
    else:
        backbone.to('cuda')
    for tag in tqdm(hashtags, total=len(hashtags)):
        if tag not in ['data', 'data backup', 'foodstagram']:
            output_path = os.path.join(path, tag, 'des/output.json')
            with open(output_path, 'rb') as f:
                output_file = json.load(f)

            for post in output_file:
                post_hashtags = []
                post_text = ''
                img_key = post['key'].split('/')[-2]
                description = post['description'].split()
                img_path = os.path.join(path, tag, 'des/' + img_key + '.jpg')
                try:
                    input_img = Image.open(img_path)
                except:
                    continue
                if torch.cuda.is_available():
                    img_tensor = img_preprocess(input_img).to("cuda")  # [3,224,224]
                else:
                    img_tensor = img_preprocess(input_img)

                # image grid 생성
                crop_idx = [q * 32 for q in range(8)]
                grid_tensor = []
                for i in range(len(crop_idx) - 1):
                    for j in range(len(crop_idx) - 1):
                        grid_tensor.append(img_tensor[:, crop_idx[i]:crop_idx[i + 1], crop_idx[j]:crop_idx[j + 1]])

                grid_tensor = torch.stack(grid_tensor)  # [7, 3, 32, 32]
                with torch.no_grad():
                    vgg_output = backbone(grid_tensor)  # [7, 1000]

                for word in description:
                    if word[0] == '@':
                        continue
                    elif word[0] == '#':
                        for thistag in word.split('#'):
                            if thistag:
                                post_hashtags.append(thistag.lower())
                    else:
                        post_text += word_process(word.lower(), use_emoji)

                pattern = '(http|ftp|https)://(?:[-\w.]|(?:\da-fA-F]{2}))+'  # url 제거
                post_text = re.sub(pattern=pattern, repl=' ', string=post_text)
                pattern = '[^\w\s]'  # 특수기호 제거
                post_text = re.sub(pattern=pattern, repl=' ', string=post_text)
                post_text = remove_space(post_text)  # 공백 제거
                if not post_text or post_text == ' ' or not post_hashtags: continue
                if post_text[0] == ' ': post_text = post_text[1:]
                hashdataset.add_data(vgg_output, post_text, post_hashtags)

    hashdataset.remove_low_freq()
    for q in range(20):
        print(hashdataset.text[q])
        print(hashdataset.hashtags[q])
        print("-------------------------------------------------")
    print(len(hashdataset.text), len(hashdataset.img), len(hashdataset.hashtags))
    print(hashdataset.word_vocab_size, hashdataset.hashtag_vocab_size)

    with open("raw_data.pkl", "wb") as f:
        pickle.dump([hashdataset.img, hashdataset.text, hashdataset.hashtags,
                     hashdataset.hashtag_vocab, hashdataset.word_vocab,
                     hashdataset.new_hashtags_onehot], f)
