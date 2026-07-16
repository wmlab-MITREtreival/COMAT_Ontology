
# COMAT: Cybersecurity Ontology for Attack Pattern Inference

## Overview
COMAT is a cybersecurity framework designed for inferring attack patterns using natural language processing techniques. This repository contains tools and models for embedding analysis, attack pattern inference, and evaluation.

## Model Downloads
Download the [Structured prediction SRL model, binary bert, and relevant ontology files](https://drive.google.com/drive/folders/1GMIK-fjkr9awD6R1DE6hl8A9uu2e8139?usp=drive_link).
After downloading, please ensure to place the models in the same directory as `main.py`.
This ensures the application can correctly locate and utilize the models.

## Database Setup (Neo4j Ontology)

The inference pipeline queries a **Neo4j** graph database that holds the COMAT
ontology (Groups, Software, Techniques, Tactics, and Verb-Object attack
patterns). You must restore this database before running `main.py`,
`predict.py`, or `attack_pattern_inference.py`.

### 1. Download the ontology dump
Download `ontology_backup_20260716.dump` from the
[model & data folder](https://drive.google.com/drive/folders/1GMIK-fjkr9awD6R1DE6hl8A9uu2e8139?usp=drive_link)
(the same Drive folder as the models above).

### 2. Restore it into Neo4j
Requires **Neo4j 4.4.x (Community Edition)**. With the Neo4j server stopped:

```bash
neo4j-admin load --from=ontology_backup_20260716.dump --database=ontology --force
```

Then set the default database in `conf/neo4j.conf`:
Start Neo4j and set a password on first login.

### 3. Configure the connection
Edit `data_source/neo4j_info.json` with your Neo4j address and credentials:

```json
{
    "url": "bolt://localhost:7687",
    "account": "neo4j",
    "password": "your_password"
}
```

All scripts read this file to connect, so this is the only place you need to
set the database address.



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
python attack_pattern_inference.py
```

