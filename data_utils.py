import json
import torch
import spacy
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any


class LegalDataLoader:
    """Handles loading and preprocessing of legal annotation data."""
    
    def __init__(self, train_path: str, dev_path: str):
        """
        Initialize data loader with file paths.
        
        Args:
            train_path (str): Path to training JSON file
            dev_path (str): Path to development JSON file
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.nlp = None
        
    def load_spacy_model(self, model_name: str = 'en_core_web_sm'):
        """Load spaCy model for text processing."""
        self.nlp = spacy.load(model_name)
        
    def load_raw_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load raw JSON data from train and dev files.
        
        Returns:
            Tuple of (train_data, dev_data) as lists of dictionaries
        """
        with open(self.train_path, 'r') as f:
            train_data = json.load(f)
        with open(self.dev_path, 'r') as f:
            dev_data = json.load(f)
        return train_data, dev_data
    
    def create_sentence_and_labels_list(self, data_raw: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extract sentences and labels from raw annotation data.
        
        Args:
            data_raw (List[Dict]): Raw annotation data
            
        Returns:
            Tuple of (sentences, labels) lists
        """
        sentences = []
        labels = []
        
        for doc in data_raw:
            annotations = doc['annotations'][0]['result']
            for annotation in annotations:
                sent = annotation['value']['text']
                label = annotation['value']['labels'][0]
                sentences.append(sent)
                labels.append(label)
        
        return sentences, labels
    
    def spacy_tokenize(self, txt: str) -> List[str]:
        """
        Tokenize text using spaCy with custom cleaning rules.
        
        Args:
            txt (str): Input text
            
        Returns:
            List of cleaned tokens
        """
        if self.nlp is None:
            raise ValueError("spaCy model not loaded. Call load_spacy_model() first.")
            
        doc = self.nlp(txt)
        clean_tokens = []
        
        for token in doc:
            if token.pos_ == 'PUNCT':
                continue
            elif token.pos_ in ['\n', '\n\n']:
                continue
            elif token.pos_ == 'NUM':
                clean_tokens.append(f'<NUM{len(token)}>')
            else:
                clean_tokens.append(token.lemma_)
                
        return clean_tokens
    
    def get_processed_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Load and process all data.
        
        Returns:
            Tuple of (train_sentences, train_labels, dev_sentences, dev_labels)
        """
        train_data, dev_data = self.load_raw_data()
        train_sentences, train_labels = self.create_sentence_and_labels_list(train_data)
        dev_sentences, dev_labels = self.create_sentence_and_labels_list(dev_data)
        
        return train_sentences, train_labels, dev_sentences, dev_labels


class Seq2SeqDataset(Dataset):
    """PyTorch Dataset for sequence-to-sequence legal text classification."""
    
    def __init__(self, sentences: List[str], labels: List[str], tokenizer):
        """
        Initialize dataset.
        
        Args:
            sentences (List[str]): Input sentences
            labels (List[str]): Target labels
            tokenizer: HuggingFace tokenizer instance
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.n_samples = len(sentences)
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            Dictionary with 'input_ids' and 'labels' keys
        """
        input_ids = self.tokenizer(self.sentences[index])['input_ids']
        label_ids = self.tokenizer(self.labels[index])['input_ids']
        
        return {
            "input_ids": input_ids,
            "labels": label_ids
        }


def create_datasets(train_path: str, dev_path: str, tokenizer) -> Tuple[Seq2SeqDataset, Seq2SeqDataset]:
    """
    Create training and development datasets.
    
    Args:
        train_path (str): Path to training data
        dev_path (str): Path to development data
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tuple of (train_dataset, dev_dataset)
    """
    data_loader = LegalDataLoader(train_path, dev_path)
    train_sentences, train_labels, dev_sentences, dev_labels = data_loader.get_processed_data()
    
    train_dataset = Seq2SeqDataset(train_sentences, train_labels, tokenizer)
    dev_dataset = Seq2SeqDataset(dev_sentences, dev_labels, tokenizer)
    
    return train_dataset, dev_dataset