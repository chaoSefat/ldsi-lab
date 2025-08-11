import torch
import json
import spacy
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from sklearn.metrics import f1_score
from data_utils import Seq2SeqDataset


class LegalSeq2SeqModel:
    """Sequence-to-sequence model for legal text classification."""
    
    def __init__(self, model_checkpoint: str = "t5-base"):
        """
        Initialize the model and tokenizer.
        
        Args:
            model_checkpoint (str): HuggingFace model checkpoint name
        """
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.trainer = None
        self.nlp = None
        
    def load_spacy_model(self, model_name: str = 'en_core_web_sm'):
        """Load spaCy model for text processing."""
        self.nlp = spacy.load(model_name)
    
    def setup_training(self, 
                      train_dataset: Seq2SeqDataset,
                      eval_dataset: Seq2SeqDataset,
                      output_dir: str = "seq2seq-legal-labeling",
                      batch_size: int = 8,
                      learning_rate: float = 4e-3,
                      num_epochs: int = 10,
                      weight_decay: float = 0.01) -> None:
        """
        Setup training configuration and trainer.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save model outputs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            weight_decay: Weight decay for regularization
        """
        model_name = self.model_checkpoint.split("/")[-1]
        
        args = Seq2SeqTrainingArguments(
            f"{model_name}-{output_dir}",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            save_total_limit=1,
            save_strategy="no",
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            push_to_hub=False,
            load_best_model_at_end=False,
        )
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        self.trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
    
    def train(self) -> None:
        """Train the model using the configured trainer."""
        if self.trainer is None:
            raise ValueError("Trainer not configured. Call setup_training() first.")
        
        self.trainer.train()
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path (str): Path to save the model
        """
        if self.trainer is None:
            raise ValueError("Trainer not configured. Call setup_training() first.")
        
        self.trainer.save_model(save_path)
    
    def spacy_tokenize(self, txt: str) -> List[str]:
        """
        Tokenize text using spaCy with custom cleaning rules.
        
        Args:
            txt (str): Input text
            
        Returns:
            List of cleaned tokens
        """
        if self.nlp is None:
            self.load_spacy_model()
            
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
    
    def generate_predictions(self, 
                           dataset: Seq2SeqDataset, 
                           device: str = 'cuda') -> List[str]:
        """
        Generate predictions for a dataset.
        
        Args:
            dataset: Dataset to generate predictions for
            device: Device to run inference on
            
        Returns:
            List of predicted labels
        """
        predictions = []
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            for i in range(len(dataset)):
                data = dataset[i]
                input_ids = torch.tensor(data['input_ids']).to(device).view(1, -1)
                
                pred = self.model.generate(input_ids=input_ids)
                pred_decoded = self.spacy_tokenize(self.tokenizer.decode(pred.squeeze(0)))
                
                # Extract clean prediction (removing special tokens)
                if len(pred_decoded) > 3:
                    pred_clean = pred_decoded[3][:-3]
                else:
                    pred_clean = pred_decoded[0] if pred_decoded else ""
                    
                predictions.append(pred_clean)
        
        return predictions
    
    def save_predictions(self, predictions: List[str], filename: str) -> None:
        """
        Save predictions to JSON file.
        
        Args:
            predictions: List of predictions
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    def evaluate_predictions(self, 
                           predictions: List[str], 
                           true_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate predictions using F1 scores.
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            
        Returns:
            Dictionary with F1 scores (weighted, macro, micro)
        """
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_micro = f1_score(true_labels, predictions, average='micro')
        
        return {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }


class ModelEvaluator:
    """Utility class for model evaluation and analysis."""
    
    @staticmethod
    def print_evaluation_results(results: Dict[str, float]) -> None:
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Dictionary with evaluation metrics
        """
        print("=== Evaluation Results ===")
        print(f"Weighted AVG F1: {results['f1_weighted']:.4f}")
        print(f"Macro AVG F1: {results['f1_macro']:.4f}")
        print(f"Micro AVG F1: {results['f1_micro']:.4f}")
    
    @staticmethod
    def save_evaluation_results(results: Dict[str, float], filename: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Dictionary with evaluation metrics
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)