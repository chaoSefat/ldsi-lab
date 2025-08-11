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