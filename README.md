
# COMAT: Cybersecurity Ontology for Attack Pattern Inference

## Overview
COMAT is a cybersecurity framework designed for inferring attack patterns using natural language processing techniques. This repository contains tools and models for embedding analysis, attack pattern inference, and evaluation.

## Installation
To run this project, make sure you have the following dependencies installed:
- Python 3.x
- Required Python packages (see requirements.txt)

You can install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Main Inference
To run the main inference process, execute the `main.py` script:
```bash
python main.py
```

This script will:
1. Load embeddings and necessary data.
2. Perform inference on the loaded data.
3. Save the prediction results to the specified directory.

### Making Predictions - For Query Node Extraction and Inferring Techniques for new input
To make predictions based on a text input, you can use the `predict.py` script:
```bash
python predict.py
```

This script processes an input document, extracts query nodes, and queries the COMAT system for attack patterns.


### `attack_pattern_inference(vo_pair_embeddings, group=[], software=[], vo_pair=[], document="", srl_info=[], tactic_list=[])`
- **Description**: Infers attack patterns based on verb-object pairs, group, and software information.
- **Returns**: A dictionary with inferred attack techniques.
```bash
python predict.py
```

