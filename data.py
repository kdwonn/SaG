from copy import deepcopy
import json
import os.path as osp
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import BertTokenizer

from matplotlib.patches import Rectangle, Polygon
from PIL import Image, ImageDraw
from multiprocessing import Manager

class ReferDatasetBert(data.Dataset):
    def __init__(self, root, splitset, max_iters=None, transform=None, crop_size=None, label_crop_size=None, scale=True, drop_prob=0):
        self.root = root
        self.train = 0
        self.crop_size = crop_size
        self.label_crop_size = label_crop_size
        self.transform = transform
        self.set = splitset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        self.drop_prob = drop_prob

        self.data_list = glob(osp.join(root, self.set+'_batch','*'))
        if not max_iters==None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            if max_iters < len(self.data_list):
                self.data_list = self.data_list[:max_iters]
        self.files = []
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sentence, img_id, image, label, im_name = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        target = process_caption_bert(sentence, self.tokenizer, self.drop_prob, self.train)
        return image, target, index, img_id
    
    def get_raw_item(self, index):
        datafiles = np.load(self.data_list[index])

        name = self.data_list[index].split('/')[-1]
        name = name.split('.')[0]

        image = Image.fromarray(datafiles["im_batch"]).convert('RGB')
        label = datafiles["mask_batch"]
        sentence = datafiles["sent_batch"][0]
        im_name= str(datafiles['im_name_batch'])
        img_id = osp.basename(self.data_list[index]).split(".")[0].split("_")[-1]
        return sentence, img_id, image, label, im_name
        

class ReferDatasetBertTexts(data.Dataset):
    def __init__(self, root, splitset, max_iters=None, transform=None, crop_size=None, 
            label_crop_size=None, scale=True, drop_prob=0, num_texts=0):
        self.root = root
        self.train = 0
        self.crop_size = crop_size
        self.label_crop_size = label_crop_size
        self.transform = transform
        self.set = splitset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab
        self.drop_prob = drop_prob
        self.num_texts = num_texts
        with open(osp.join(osp.dirname(root), '%s_imgtotxts.json' % osp.basename(root))) as f:
            self.img_txts_dict = json.load(f)
        with open(osp.join(osp.dirname(root), '%s_imgtoname.json' % osp.basename(root))) as fp:
            self.img_im_name_dict = json.load(fp)

        # self.data_list = glob(osp.join(root, self.set+'_batch','*'))
        # if not max_iters==None:
        #     self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        #     if max_iters < len(self.data_list):
        #         self.data_list = self.data_list[:max_iters]

        self.data_list = [key for key in self.img_im_name_dict.keys()]
        if not max_iters==None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            if max_iters < len(self.data_list):
                self.data_list = self.data_list[:max_iters]

        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sentences, img_id, image, label, _, num_sent = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.num_texts is not 0:
            sentences = random.sample(sentences, self.num_texts) if len(sentences) > self.num_texts else sentences
        targets = [process_caption_bert(s, self.tokenizer, self.drop_prob, self.train) for s in sentences]
        return image, targets, index, img_id
    
    def get_raw_item(self, index):
        img_id = self.img_im_name_dict[self.data_list[index]][0]
        datafiles = np.load(osp.join(self.root, self.set+'_batch', img_id))

        # name = self.data_list[index].split('/')[-1]
        # name = name.split('.')[0]

        image = Image.fromarray(datafiles["im_batch"]).convert('RGB')
        label = datafiles["mask_batch"]
        sentence = datafiles["sent_batch"][0]
        img_id = img_id.split(".")[0].split("_")[-1]
        im_name = str(datafiles['im_name_batch'])
        sentences = self.img_txts_dict[im_name]
        num_sent = len(sentences)
        
        return sentences, img_id, image, label, im_name, num_sent


class PhraseDatasetBert(data.Dataset):
    def __init__(self, root, splitset, max_iters=None, transform=None, crop_size=None, label_crop_size=None, scale=True, drop_prob=0):
        self.root = root
        self.train = 0
        self.crop_size = crop_size
        self.label_crop_size = label_crop_size
        self.transform = transform
        self.set = splitset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.vocab = self.tokenizer.vocab
        self.drop_prob = drop_prob
        self.data_list = []
        with open(osp.join("./data/phrasecut/", "refer_%s_ris.json") % splitset, "r") as json_file:
            ref_tasks_dict = json.load(json_file)
            for i, (key, values) in enumerate(ref_tasks_dict.items()): 
                if splitset == 'train':
                    self.data_list.append((key, values['phrase'], None))
                elif splitset == 'val':
                    image = Image.open(osp.join(root,'images', key.split('__')[0]+'.jpg')).convert('RGB')
                    polygons = []
                    for ps in values['Polygons']:
                        polygons += ps
                    label = polygons_to_mask(polygons, image.size[0], image.size[1])
                    self.data_list.append((key, values['phrase'], label))
                    # del label, image, polygons
                if not max_iters == None:
                    if i > max_iters-1:
                        break
            del ref_tasks_dict, json_file
                
        self.data_ids = [i for i in range(len(self.data_list))]

        manager = Manager()
        self.data_list = manager.list(self.data_list)
        # self.data_ids = manager.list(self.data_ids)
        
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sentence, img_id, image, label, _ = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        target = process_caption_bert(sentence, self.tokenizer, self.drop_prob, self.train)

        del label, sentence
        return image, target, index, img_id
    
    
    def get_raw_item(self, index):
        ref_id, sentence, label = self.data_list[self.data_ids[index]]

        image = Image.open(osp.join(self.root,'images', ref_id.split('__')[0]+'.jpg')).convert('RGB')
        
        return sentence, ref_id, image, label, None

def polygons_to_mask(polygons, w, h):
    p_mask = np.zeros((h, w))
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        p = []
        for x, y in polygon:
            p.append((int(x), int(y)))
        # img = Image.new('L', (w, h), 0)
        with Image.new('L', (w, h), 0) as im:
            ImageDraw.Draw(im).polygon(p, outline=1, fill=1)
            mask = np.array(im)
        p_mask += mask
    del im, mask, p
    p_mask = p_mask > 0
    return p_mask

def process_caption_bert(caption, tokenizer, drop_prob, train):
        output_tokens = []
        deleted_idx = []
        tokens = tokenizer.basic_tokenizer.tokenize(caption)
        
        for i, token in enumerate(tokens):
            sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()

            if prob < drop_prob and train:  # mask/remove the tokens only during training
                prob /= drop_prob

                # 50% randomly change token to mask token
                if prob < 0.5:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.6:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        deleted_idx.append(len(output_tokens) - 1)
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        target = tokenizer.convert_tokens_to_ids(output_tokens)
        target = torch.Tensor(target)
        return target


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
        data: list of (image, sentence) tuple.
            - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
            - sentence: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256) or 
                        (batch_size, padded_length, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = torch.tensor([len(cap) for cap in sentences])
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, cap_lengths, ids


def collate_fn_fast(data):
    """
        input : List of tuples. Each tuple is a output of __getitem__ of the dataset
        output : Collated tensor
    """
    # Sort a data list by sentence length
    images, sentences, img_ids, _ = zip(*data)
    # image, sentences, index, img_id
    # compute the number of captions in each images and create match label from it
    flatten_sentences = [sentence for img in list(sentences) for sentence in img]
    flatten_sentences_len = [len(sentence) for sentence in flatten_sentences]
    n_sents_for_img = [len(sents) for sents in list(sentences)]             #len = batch

    org_len, org_sen = flatten_sentences_len, flatten_sentences
    caption_data = list(zip(flatten_sentences_len, flatten_sentences))
    sorted_idx = sorted(range(len(caption_data)), key=lambda x: caption_data[x][0], reverse=True)
    recovery_idx = sorted(range(len(caption_data)), key=lambda x: sorted_idx[x], reverse=False)
    caption_data.sort(key=lambda x: x[0], reverse=True)
    flatten_sentences_len, flatten_sentences = zip(*caption_data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    sentences_len = torch.tensor(flatten_sentences_len)
    recovery_idx = torch.tensor(recovery_idx)
    n_sents_for_img = torch.tensor(n_sents_for_img)
    padded_sentences = torch.zeros(len(flatten_sentences), max(sentences_len)).long()
    for i, cap in enumerate(flatten_sentences):
        end = sentences_len[i]
        padded_sentences[i, :end] = cap[:end]
    return images, padded_sentences, sentences_len, recovery_idx, n_sents_for_img, img_ids


def get_loader_single(
    data_name, split, root, transform, fast_batch,
    batch_size=128, shuffle=True, num_workers=2, max_iters=None, num_texts=0):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if 'coco' == data_name:
        if fast_batch:
            dataset = ReferDatasetBertTexts(
            root=root,
            splitset = split,
            transform=transform,
            num_texts=num_texts)
        else:
            dataset = ReferDatasetBert(
                root=root,
                splitset = split,
                transform=transform)
    elif 'phrasecut' == data_name:
        dataset = PhraseDatasetBert(
            root=root,
            splitset = split,
            transform=transform,
            max_iters=max_iters
        )
    else:
        assert NotImplementedError
    
    collate = collate_fn_fast if fast_batch else collate_fn

    # Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        # prefetch_factor = 2,
        persistent_workers=True,
        collate_fn=collate)
    
    return data_loader


def get_image_transform(split_name, img_backbone, crop_size, use_aug):
    if 'vit' in img_backbone:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif 'res' in img_backbone:
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError
    
    t_list = []
    if split_name == 'train':
        if use_aug:
            t_list = [
                transforms.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                # transforms.RandomHorizontalFlip()
            ]
        else:
            t_list = [transforms.Resize(size=(crop_size, crop_size))]
    elif split_name == 'val':
        t_list = [
            transforms.Resize(size=(crop_size, crop_size))
        ]
    t_end = [
        transforms.ToTensor(), 
        normalizer
    ]
    
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_train_loader(args):
    transform = get_image_transform('train', args.img_backbone, args.crop_size, args.use_aug)
    return get_loader_single(
        data_name=args.data_name, 
        split=args.data_split,
        root=args.data_path,
        transform=transform,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers,
        fast_batch=args.fast_batch,
        num_texts=args.num_texts,
    )


def get_test_loader(args):
    transform = get_image_transform('val', args.img_backbone, args.crop_size, args.use_aug)
    return get_loader_single(
        data_name=args.data_name, 
        split='val',
        root=args.data_path,
        transform=transform,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=1,
        max_iters = 10000,
        fast_batch=False,
        num_texts=args.num_texts,
    )


def get_train_pseudo_loader(args):
    transform = get_image_transform('train', args.img_backbone, args.crop_size, use_aug=False)
    return get_loader_single(
        data_name=args.data_name, 
        split=args.data_split,
        root=args.data_path,
        transform=transform,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=1,
        fast_batch=False,
        num_texts=args.num_texts,
    )