# enzparampip
 Automated protein/substrate Kinetic Parameter Pipeline
# Enzyme Parameter Prediction Pipeline

1. Take a metabolic network as input
2. Parse all enzymes in the network
3. For each enzyme:
   - Retrieve protein sequences from UniProt
   - Get small molecule SMILES representations
   - Predict enzyme kinetic parameters (kcat and km)
4. Update the metabolic network with the predicted parameters

I've created a comprehensive end-to-end pipeline for automating enzyme parameter prediction as requested. Let me explain the key components and how to use this solution:

## Pipeline Overview

The code implements a pipeline that:

1. Takes a metabolic network as input (in various formats)
2. For each enzyme in the network:
   - Automatically retrieves protein sequences from UniProt
   - Gets SMILES representations for substrates
   - Predicts kinetic parameters (kcat and Km) using ESM protein embeddings
3. Updates the network with predicted parameters
4. Outputs the enriched network in your preferred format

## Key Features

1. **Flexible Input/Output**:
   - Supports SBML, JSON, and CSV formats
   - Automatically handles format conversion

2. **Automated Data Retrieval**:
   - UniProt API integration for protein sequences
   - PubChem/KEGG integration for chemical structures
   - Caching to minimize redundant API calls

3. **Parameter Prediction**:
   - Uses ESM protein language model for embeddings
   - Integrates with RDKit for molecular fingerprints
   - Currently uses placeholder predictions (would need actual fine-tuned model)

4. **Command-line Interface**:
   - Main `run` command for pipeline execution
   - `sample` command to generate example networks
   - `convert` command for format conversion

## Usage Examples

### Basic Usage

```bash
# Run the pipeline on a network file
python enzyme_parameter_pipeline.py run my_network.json -o enriched_network.json

# Generate a sample network file
python enzyme_parameter_pipeline.py sample sample_network.json

# Convert between formats
python enzyme_parameter_pipeline.py convert network.json network.csv
```

### Programmatic Usage

```python
from enzyme_parameter_pipeline import EnzymeParameterPipeline, NetworkParser

# Load network
reactions = NetworkParser.parse("my_network.json")

# Run pipeline
pipeline = EnzymeParameterPipeline()
enriched_reactions = pipeline.run("my_network.json")

# Access predicted parameters
for reaction in enriched_reactions:
    if reaction.kcat:
        print(f"Reaction {reaction.id}: kcat = {reaction.kcat} /s")
        for substrate_id, km in reaction.km.items():
            print(f"  Km for {substrate_id} = {km} mM")
```

## Implementation Notes

1. The current code uses a placeholder prediction model. For production use, you would need to:
   - Train/fine-tune an actual model on enzyme-substrate data
   - Replace the placeholder prediction in the `ESMParameterPredictor` class

2. The pipeline is extensible:
   - Add more database connectors as needed
   - Support additional file formats
   - Implement more sophisticated prediction algorithms

3. Error handling and caching are implemented for robust operation with external APIs

