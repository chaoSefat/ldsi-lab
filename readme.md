# Legal Rhetorical Role Classification using Sequence-to-Sequence Models

## Context
This implementation was developed as part of the **Legal Data Science Lab (LDSI_LAB)** at **Technical University of Munich (TUM)** for the Master's in Informatics program (SS22). The work explores innovative approaches to legal NLP by applying sequence-to-sequence models to the structured analysis of Indian legal judgment documents, contributing to the broader effort of automating legal document processing in populous judicial systems.

## Overview
This module implements a sequence-to-sequence approach for automatic rhetorical role classification in Indian legal judgment documents. The system uses transformer models (T5-base) to predict rhetorical roles for individual sentences in legal documents, treating the classification task as a text generation problem where the model learns to generate appropriate rhetorical role labels given input legal text segments.

## Rhetorical Roles in Legal Documents
Legal judgment documents can be segmented into topically coherent semantic units called Rhetorical Roles (RRs). These roles help structure legal documents for better organization, search, and automated processing.
Dataset Details

- Source: Indian Supreme Court judgment documents in English
- Corpus Size: 354 legal documents with 40,305 annotated sentences
- Annotation: 12 different rhetorical role categories
- Granularity: Sentence-level annotations
- Origin: Part of the BUILDNyAI project for legal NLP corpus development

## The 12 Rhetorical Role Categories:

- Preamble: Document header and case identification
- Facts: Statement of facts and case background
- Arguments: Legal arguments presented by parties
- Statute: Referenced laws and statutory provisions
- Precedent: Cited previous court decisions
- Ratio: Court's reasoning and legal principles
- Ruling: Final decision and orders
- Dissent: Dissenting opinions (if any)
- Concurrence: Concurring opinions
- Analysis: Court's analysis of law and facts
- Issues: Legal issues identified by the court
- Other: Miscellaneous content not fitting other categories


## Technical Approach
## Sequence-to-Sequence Framework
Unlike traditional classification approaches, this implementation treats rhetorical role prediction as a text generation task:

- Input: Legal sentence text (tokenized)
- Processing: T5 encoder-decoder architecture
- Output: Generated rhetorical role label
- Advantage: Leverages pre-trained language model's understanding of legal language

## Model Architecture

- Base Model: T5-base (Text-to-Text Transfer Transformer)
- Task Formulation: Sentence → Rhetorical Role Label generation
- Preprocessing: Custom spaCy-based tokenization for legal text
- Post-processing: Label extraction and cleaning

## Features

- Domain-Specific Design: Tailored for Indian legal judgment documents
- Sentence-Level Classification: Processes individual sentences for fine-grained analysis
- Custom Legal Text Preprocessing: Handles legal document peculiarities
- Comprehensive Evaluation: Multi-metric assessment (weighted, macro, micro F1-scores)
- Modular Architecture: Clean separation of data handling, model operations, and training
- Flexible Configuration: Adjustable hyperparameters and model checkpoints

## Results \& Findings:
### Model Performance
We evaluated the T5 sequence-to-sequence model on multiple datasets to assess its effectiveness for rhetorical role classification in legal documents. The model was trained with limited computational resources (Google Colab) with linmited training time on free plan, constraining the number of training epochs.
### Performance on Legal Datasets
### Kalamkar et al. Dataset:

- Training Duration: 5 epochs
- Accuracy: 0.503
- Macro F1: 0.219
- Weighted F1: 0.448

### Bhattacharya et al. Dataset:

- Training Duration: 6 epochs
- Accuracy: 0.363
- Macro F1: 0.246
- Weighted F1: 0.356

### Comparative Analysis:
The T5 sequence-to-sequence model demonstrated notably better performance on the Kalamkar dataset compared to the Bhattacharya dataset across all evaluation metrics. The Kalamkar dataset yielded approximately 39% higher accuracy (0.503 vs 0.363) and 26% higher weighted F1-score (0.448 vs 0.356).
### Impact of Dataset Size
The performance disparity between datasets can be attributed primarily to the significant difference in training data availability. The Bhattacharya dataset contains only 50 documents, which is substantially smaller compared to the Kalamkar dataset. This limited training data severely constrains the model's ability to learn effective representations for rhetorical role classification.
To validate this hypothesis, we conducted additional experiments on the PubMed20k RCT dataset, which contains a much larger corpus for sequential sentence classification:
PubMed20k RCT Results:

- Accuracy: 0.752
- Macro F1: 0.682
- Weighted F1: 0.747

The superior performance on PubMed20k RCT (49% improvement in accuracy over Kalamkar and 107% improvement over Bhattacharya) strongly corroborates that dataset size is a critical factor in the T5 model's performance for sequential sentence classification tasks.

### Limitations and Constraints
The experimental results should be interpreted with the following limitations in mind:

1. Limited Training Epochs: Due to computational resource constraints, models were trained for only 5-6 epochs, which may be insufficient for full convergence.
2. Hyperparameter Optimization: The scope of hyperparameter tuning was restricted due to high computational requirements and limited compute provisioning.
3. Early Stopping: Training was terminated early due to resource usage limits, potentially preventing the models from reaching optimal performance.

### Key Findings:

- Dataset Size Dependency: The T5 sequence-to-sequence approach shows strong sensitivity to training data size, with performance scaling significantly with larger datasets.
- Legal Domain Challenges: Performance on legal datasets (Kalamkar and Bhattacharya) was notably lower than on biomedical abstracts (PubMed20k), suggesting domain-specific challenges in legal text processing.
- Resource Requirements: The high computational demands of T5-based sequence-to-sequence models present practical constraints for legal NLP applications with limited resources.