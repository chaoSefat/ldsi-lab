import argparse
import os
from typing import Optional
from data_utils import create_datasets, LegalDataLoader
from model_utils import LegalSeq2SeqModel, ModelEvaluator


def main(args: argparse.Namespace) -> None:
    """
    Main training and evaluation pipeline.
    
    Args:
        args: Command line arguments
    """
    print("=== Legal Text Seq2Seq Classification ===")
    print(f"Model: {args.model_checkpoint}")
    print(f"Training data: {args.train_path}")
    print(f"Development data: {args.dev_path}")
    
    # Initialize model
    print("\n1. Initializing model and tokenizer...")
    model = LegalSeq2SeqModel(model_checkpoint=args.model_checkpoint)
    
    # Load and create datasets
    print("2. Loading and preprocessing data...")
    train_dataset, dev_dataset = create_datasets(
        args.train_path, 
        args.dev_path, 
        model.tokenizer
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Development samples: {len(dev_dataset)}")
    
    # Setup training
    print("3. Setting up training configuration...")
    model.setup_training(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("4. Starting training...")
    model.train()
    
    # Save model
    if args.save_model_path:
        print(f"5. Saving model to {args.save_model_path}...")
        model.save_model(args.save_model_path)
    
    # Generate predictions and evaluate
    if args.evaluate:
        print("6. Generating predictions and evaluating...")
        
        # Load original labels for evaluation
        data_loader = LegalDataLoader(args.train_path, args.dev_path)
        _, _, _, dev_labels = data_loader.get_processed_data()
        
        # Generate predictions
        predictions = model.generate_predictions(dev_dataset, device=args.device)
        
        # Save predictions
        pred_filename = f"predictions_{args.model_checkpoint.split('/')[-1]}_{args.batch_size}_{args.num_epochs}.json"
        model.save_predictions(predictions, pred_filename)
        print(f"Predictions saved to: {pred_filename}")
        
        # Evaluate
        results = model.evaluate_predictions(predictions, dev_labels)
        ModelEvaluator.print_evaluation_results(results)
        
        # Save evaluation results
        eval_filename = f"evaluation_{args.model_checkpoint.split('/')[-1]}_{args.batch_size}_{args.num_epochs}.json"
        ModelEvaluator.save_evaluation_results(results, eval_filename)
        print(f"Evaluation results saved to: {eval_filename}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Legal Text Sequence-to-Sequence Classification"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_path", 
        type=str, 
        default="train.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--dev_path", 
        type=str, 
        default="dev.json",
        help="Path to development data JSON file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        default="t5-base",
        help="HuggingFace model checkpoint"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=4e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay for regularization"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="seq2seq-legal-labeling",
        help="Output directory for training artifacts"
    )
    parser.add_argument(
        "--save_model_path", 
        type=str, 
        default=None,
        help="Path to save the trained model"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training and inference"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    import torch
    
    args = parse_arguments()
    
    # Validate file paths
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Training data file not found: {args.train_path}")
    if not os.path.exists(args.dev_path):
        raise FileNotFoundError(f"Development data file not found: {args.dev_path}")
    
    main(args)