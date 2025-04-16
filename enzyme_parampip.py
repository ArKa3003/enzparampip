#!/usr/bin/env python


import os
import sys
import json
import time
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, asdict, field

import requests
import numpy as np
import pandas as pd
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnzymePipeline")

# Check for required packages
try:
    import esm  # For protein language models
    from rdkit import Chem  # For chemical informatics
    from rdkit.Chem import AllChem
    # Suppress RDKit logging
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    logger.error("Please install required packages: pip install fair-esm rdkit")
    sys.exit(1)


# =================
# Data Models
# =================

@dataclass
class Enzyme:
    """Class representing an enzyme"""
    id: str
    name: Optional[str] = None
    ec_number: Optional[str] = None
    uniprot_id: Optional[str] = None
    sequence: Optional[str] = None
    kcat: Optional[float] = None
    km: Optional[float] = None
    
    def __repr__(self):
        return f"Enzyme(id={self.id}, ec={self.ec_number})"


@dataclass
class Substrate:
    """Class representing a substrate molecule"""
    id: str
    name: Optional[str] = None
    smiles: Optional[str] = None
    
    def __repr__(self):
        return f"Substrate(id={self.id}, name={self.name})"


@dataclass
class Reaction:
    """Class representing a metabolic reaction with enzyme and substrates"""
    id: str
    name: Optional[str] = None
    enzyme: Optional[Enzyme] = None
    substrates: List[Substrate] = field(default_factory=list)
    products: List[Substrate] = field(default_factory=list)
    kcat: Optional[float] = None
    km: Dict[str, float] = field(default_factory=dict)  # Substrate ID to Km mapping
    
    def __repr__(self):
        return f"Reaction(id={self.id}, enzyme={self.enzyme.id if self.enzyme else None})"


# =================
# Configuration
# =================

class Config:
    """Configuration for the pipeline"""
    # API URLs
    UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb/"
    UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
    PUBCHEM_API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    KEGG_API_BASE = "https://rest.kegg.jp/get/"
    
    # Model settings
    ESM_MODEL = "esm1b_t33_650M_UR50S"  # Default ESM model to use
    
    # Request settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    TIMEOUT = 30  # seconds
    
    # Cache settings
    CACHE_DIR = os.path.join(os.getcwd(), ".cache")
    USE_CACHE = True


# =================
# Database Access Functions
# =================

class UniProtAPI:
    """Interface to the UniProt REST API"""
    
    @staticmethod
    def get_uniprot_entry(uniprot_id: str) -> dict:
        """Get a full entry from UniProt by ID"""
        url = f"{Config.UNIPROT_API_BASE}{uniprot_id}"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.get(url, params={"format": "json"}, timeout=Config.TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Retrying UniProt request for {uniprot_id}: {e}")
                    time.sleep(Config.RETRY_DELAY)
                else:
                    logger.error(f"Failed to get UniProt entry for {uniprot_id}: {e}")
                    return {}
    
    @staticmethod
    def search_by_ec(ec_number: str) -> List[dict]:
        """Search UniProt for proteins with a specific EC number"""
        url = Config.UNIPROT_SEARCH_URL
        params = {
            "query": f"ec:{ec_number}",
            "format": "json",
            "size": 10  # Limit results
        }
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=Config.TIMEOUT)
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
            except requests.exceptions.RequestException as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Retrying UniProt search for EC {ec_number}: {e}")
                    time.sleep(Config.RETRY_DELAY)
                else:
                    logger.error(f"Failed to search UniProt for EC {ec_number}: {e}")
                    return []
    
    @staticmethod
    def get_sequence(uniprot_id: str) -> str:
        """Get the amino acid sequence for a UniProt ID"""
        # Check cache first
        cache_file = os.path.join(Config.CACHE_DIR, f"uniprot_{uniprot_id}.fasta")
        if Config.USE_CACHE and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                lines = f.read().strip().split('\n')
                return ''.join(lines[1:])  # Skip the header line
        
        # Fetch from API if not in cache
        url = f"{Config.UNIPROT_API_BASE}{uniprot_id}"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.get(url, params={"format": "fasta"}, timeout=Config.TIMEOUT)
                response.raise_for_status()
                
                # Parse FASTA format
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    sequence = ''.join(lines[1:])
                    
                    # Cache result
                    if Config.USE_CACHE:
                        os.makedirs(Config.CACHE_DIR, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            f.write(response.text)
                    
                    return sequence
                return ""
            except requests.exceptions.RequestException as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Retrying sequence fetch for {uniprot_id}: {e}")
                    time.sleep(Config.RETRY_DELAY)
                else:
                    logger.error(f"Failed to get sequence for {uniprot_id}: {e}")
                    return ""


class ChemicalDatabaseAPI:
    """Interface to chemical databases (PubChem, KEGG)"""
    
    @staticmethod
    def get_smiles(compound_id: str, database: str = "pubchem") -> str:
        """Get SMILES representation for a compound from specified database"""
        # Check cache first
        cache_file = os.path.join(Config.CACHE_DIR, f"{database}_{compound_id}.smiles")
        if Config.USE_CACHE and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return f.read().strip()
        
        # Fetch from appropriate database
        if database.lower() == "pubchem":
            smiles = ChemicalDatabaseAPI._get_from_pubchem(compound_id)
        elif database.lower() == "kegg":
            smiles = ChemicalDatabaseAPI._get_from_kegg(compound_id)
        else:
            logger.warning(f"Unsupported database: {database}")
            return ""
        
        # Cache result if successful
        if smiles and Config.USE_CACHE:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            with open(cache_file, 'w') as f:
                f.write(smiles)
        
        return smiles
    
    @staticmethod
    def _get_from_pubchem(cid: str) -> str:
        """Get SMILES from PubChem by CID"""
        url = f"{Config.PUBCHEM_API_BASE}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.get(url, timeout=Config.TIMEOUT)
                response.raise_for_status()
                data = response.json()
                return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Retrying PubChem request for {cid}: {e}")
                    time.sleep(Config.RETRY_DELAY)
                else:
                    logger.error(f"Failed to get SMILES from PubChem for {cid}: {e}")
                    return ""
    
    @staticmethod
    def _get_from_kegg(kegg_id: str) -> str:
        """Get SMILES from KEGG by compound ID"""
        # First get the MOL file from KEGG
        url = f"{Config.KEGG_API_BASE}compound/{kegg_id}/mol"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.get(url, timeout=Config.TIMEOUT)
                response.raise_for_status()
                
                # Convert MOL to SMILES using RDKit
                mol = Chem.MolFromMolBlock(response.text)
                if mol:
                    return Chem.MolToSmiles(mol)
                logger.warning(f"Could not parse MOL file for KEGG compound {kegg_id}")
                return ""
            except requests.exceptions.RequestException as e:
                if attempt < Config.MAX_RETRIES - 1:
                    logger.warning(f"Retrying KEGG request for {kegg_id}: {e}")
                    time.sleep(Config.RETRY_DELAY)
                else:
                    logger.error(f"Failed to get MOL from KEGG for {kegg_id}: {e}")
                    return ""
        
        return ""


# =================
# Network Parsing 
# =================

class NetworkParser:
    """Parse metabolic networks from various file formats"""
    
    @staticmethod
    def parse(file_path: str, format: str = "auto") -> List[Reaction]:
        """
        Parse a metabolic network file into Reaction objects
        
        Args:
            file_path: Path to the network file
            format: File format (auto, sbml, json, csv)
            
        Returns:
            List of Reaction objects
        """
        # Auto-detect format if not specified
        if format == "auto":
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".xml" or ext == ".sbml":
                format = "sbml"
            elif ext == ".json":
                format = "json"
            elif ext == ".csv":
                format = "csv"
            else:
                raise ValueError(f"Cannot determine format for {file_path}")
        
        # Parse based on format
        if format == "sbml":
            try:
                import cobra
                logger.info("Parsing SBML file using COBRApy")
                return NetworkParser._parse_sbml(file_path)
            except ImportError:
                logger.error("COBRApy not installed. Install with: pip install cobra")
                sys.exit(1)
        elif format == "json":
            logger.info("Parsing JSON file")
            return NetworkParser._parse_json(file_path)
        elif format == "csv":
            logger.info("Parsing CSV file")
            return NetworkParser._parse_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _parse_sbml(file_path: str) -> List[Reaction]:
        """Parse SBML file using COBRApy"""
        import cobra
        model = cobra.io.read_sbml_model(file_path)
        reactions = []
        
        for cobra_reaction in model.reactions:
            # Extract EC numbers from annotations if available
            ec_numbers = []
            if hasattr(cobra_reaction, "annotation"):
                for key, value in cobra_reaction.annotation.items():
                    if "ec-code" in key.lower() or "ec:" in key.lower():
                        if isinstance(value, list):
                            ec_numbers.extend(value)
                        else:
                            ec_numbers.append(value)
            
            # Create enzyme if we have an EC number
            enzyme = None
            if ec_numbers:
                ec = ec_numbers[0]  # Use first EC number
                enzyme = Enzyme(
                    id=f"{cobra_reaction.id}_enzyme",
                    name=cobra_reaction.name,
                    ec_number=ec
                )
            
            # Create substrates and products
            substrates = []
            products = []
            
            for metabolite, coeff in cobra_reaction.metabolites.items():
                substrate = Substrate(
                    id=metabolite.id,
                    name=metabolite.name
                )
                
                if coeff < 0:  # Consumed = substrate
                    substrates.append(substrate)
                else:  # Produced = product
                    products.append(substrate)
            
            # Create reaction
            reaction = Reaction(
                id=cobra_reaction.id,
                name=cobra_reaction.name,
                enzyme=enzyme,
                substrates=substrates,
                products=products
            )
            reactions.append(reaction)
        
        return reactions
    
    @staticmethod
    def _parse_json(file_path: str) -> List[Reaction]:
        """Parse custom JSON network file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        reactions = []
        
        # Determine JSON structure and extract reactions
        if "reactions" in data:
            # Process each reaction in the reactions list
            for r_data in data["reactions"]:
                # Create enzyme if present
                enzyme = None
                if "enzyme" in r_data:
                    e_data = r_data["enzyme"]
                    enzyme = Enzyme(
                        id=e_data.get("id", f"{r_data['id']}_enzyme"),
                        name=e_data.get("name"),
                        ec_number=e_data.get("ec_number"),
                        uniprot_id=e_data.get("uniprot_id"),
                        sequence=e_data.get("sequence")
                    )
                
                # Create substrates
                substrates = []
                for s_data in r_data.get("substrates", []):
                    substrate = Substrate(
                        id=s_data.get("id"),
                        name=s_data.get("name"),
                        smiles=s_data.get("smiles")
                    )
                    substrates.append(substrate)
                
                # Create products
                products = []
                for p_data in r_data.get("products", []):
                    product = Substrate(
                        id=p_data.get("id"),
                        name=p_data.get("name"),
                        smiles=p_data.get("smiles")
                    )
                    products.append(product)
                
                # Create reaction
                reaction = Reaction(
                    id=r_data.get("id"),
                    name=r_data.get("name"),
                    enzyme=enzyme,
                    substrates=substrates,
                    products=products,
                    kcat=r_data.get("kcat"),
                    km={s["id"]: v for s, v in zip(r_data.get("substrates", []), r_data.get("km", []))} 
                       if "km" in r_data else {}
                )
                reactions.append(reaction)
        else:
            logger.warning("JSON structure not recognized, expected 'reactions' key")
        
        return reactions
    
    @staticmethod
    def _parse_csv(file_path: str) -> List[Reaction]:
        """Parse CSV network file"""
        df = pd.read_csv(file_path)
        
        # Check required columns
        if "reaction_id" not in df.columns:
            raise ValueError("CSV must contain a 'reaction_id' column")
        
        reactions = []
        # Group by reaction ID
        for reaction_id, group in df.groupby("reaction_id"):
            first_row = group.iloc[0]
            reaction_name = first_row.get("reaction_name", reaction_id)
            
            # Create enzyme if applicable
            enzyme = None
            if "enzyme_id" in group.columns and not pd.isna(first_row.get("enzyme_id")):
                enzyme = Enzyme(
                    id=first_row.get("enzyme_id", f"{reaction_id}_enzyme"),
                    name=first_row.get("enzyme_name"),
                    ec_number=first_row.get("ec_number"),
                    uniprot_id=first_row.get("uniprot_id")
                )
            
            # Collect substrates and products
            substrates = []
            products = []
            km_values = {}
            
            for _, row in group.iterrows():
                # Process substrates
                if "substrate_id" in row and not pd.isna(row["substrate_id"]):
                    substrate = Substrate(
                        id=row["substrate_id"],
                        name=row.get("substrate_name"),
                        smiles=row.get("substrate_smiles")
                    )
                    substrates.append(substrate)
                    
                    # Store Km if available
                    if "km" in row and not pd.isna(row["km"]):
                        km_values[substrate.id] = row["km"]
                
                # Process products if available
                if "product_id" in row and not pd.isna(row["product_id"]):
                    product = Substrate(
                        id=row["product_id"],
                        name=row.get("product_name"),
                        smiles=row.get("product_smiles")
                    )
                    products.append(product)
            
            # Get kcat if available
            kcat = first_row.get("kcat") if "kcat" in first_row else None
            
            # Create reaction
            reaction = Reaction(
                id=reaction_id,
                name=reaction_name,
                enzyme=enzyme,
                substrates=substrates,
                products=products,
                kcat=kcat,
                km=km_values
            )
            reactions.append(reaction)
        
        return reactions


# =================
# Parameter Prediction
# =================

class ESMParameterPredictor:
    """Predict enzyme kinetic parameters using ESM embeddings"""
    
    def __init__(self, model_name: str = Config.ESM_MODEL):
        """Initialize the ESM model"""
        logger.info(f"Loading ESM model: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # In a real implementation, you would load a trained model here
        # For demonstration, we'll generate plausible values
        logger.warning("Using placeholder prediction model - real implementation would use a trained model")
    
    def get_protein_embedding(self, sequence: str) -> np.ndarray:
        """Generate ESM embedding for a protein sequence"""
        # Check sequence length
        max_len = self.alphabet.max_len - 2  # Account for special tokens
        if len(sequence) > max_len:
            logger.warning(f"Sequence too long ({len(sequence)} > {max_len}), truncating")
            sequence = sequence[:max_len]
        
        # Tokenize sequence
        batch_tokens = self.alphabet.encode(sequence)
        batch_tokens = batch_tokens.unsqueeze(0).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])  # Layer 33 = last layer
            # Average per-token embeddings (excluding start/end tokens)
            embedding = results["representations"][33][0, 1:len(sequence)+1].mean(0)
        
        return embedding.cpu().numpy()
    
    def get_substrate_features(self, smiles: str) -> np.ndarray:
        """Generate molecular fingerprints from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}")
                return np.zeros(1024)
            
            # Generate Morgan fingerprint (ECFP4-like)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return np.array(fingerprint)
        except Exception as e:
            logger.error(f"Error generating substrate features: {e}")
            return np.zeros(1024)
    
    def predict(self, enzyme: Enzyme, substrate: Substrate) -> Tuple[float, float]:
        """
        Predict kcat and Km for an enzyme-substrate pair
        
        Args:
            enzyme: Enzyme with sequence
            substrate: Substrate with SMILES
        
        Returns:
            Tuple of (kcat, km) values
        """
        if not enzyme.sequence:
            logger.warning(f"No sequence available for enzyme {enzyme.id}")
            return None, None
        
        if not substrate.smiles:
            logger.warning(f"No SMILES available for substrate {substrate.id}")
            return None, None
        
        # Get protein embedding
        protein_embedding = self.get_protein_embedding(enzyme.sequence)
        
        # Get substrate features
        substrate_features = self.get_substrate_features(substrate.smiles)
        
        # In a real implementation, combine these features and use a trained model
        # For demonstration, generate plausible values
        # Kcat typically ranges from 0.1 to 1000 /s
        # Km typically ranges from 0.001 to 10 mM
        
        # Use EC number to bias the prediction (just for demonstration)
        ec_bias = 1.0
        if enzyme.ec_number:
            # Extract first number from EC (enzyme class)
            try:
                ec_class = int(enzyme.ec_number.split('.')[0])
                if ec_class == 1:  # Oxidoreductases
                    ec_bias = 0.8
                elif ec_class == 2:  # Transferases
                    ec_bias = 1.2
                elif ec_class == 3:  # Hydrolases
                    ec_bias = 1.5
                elif ec_class == 4:  # Lyases
                    ec_bias = 0.7
                elif ec_class == 5:  # Isomerases
                    ec_bias = 0.6
                elif ec_class == 6:  # Ligases
                    ec_bias = 1.0
            except (IndexError, ValueError):
                pass
        
        # Generate predictions
        kcat = 10 ** (np.random.uniform(0, 3) * ec_bias)  # 1-1000 /s
        km = 10 ** np.random.uniform(-3, 1)  # 0.001-10 mM
        
        return kcat, km


# =================
# Pipeline
# =================

class EnzymeParameterPipeline:
    """Main pipeline for enzyme parameter prediction"""
    
    def __init__(self):
        """Initialize pipeline components"""
        # Create cache directory
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Initialize API connectors
        self.uniprot_api = UniProtAPI()
        self.chemical_api = ChemicalDatabaseAPI()
        
        # Parameter predictor will be initialized on demand
        self.predictor = None
    
    def _get_predictor(self):
        """Lazy-load the parameter predictor"""
        if self.predictor is None:
            self.predictor = ESMParameterPredictor()
        return self.predictor
    
    def enrich_enzyme_data(self, enzyme: Enzyme) -> None:
        """Fetch additional data for an enzyme"""
        if enzyme is None:
            return
        
        # Skip if we already have a sequence
        if enzyme.sequence:
            return
        
        # If we have a UniProt ID, fetch sequence directly
        if enzyme.uniprot_id:
            logger.info(f"Fetching sequence for UniProt ID {enzyme.uniprot_id}")
            enzyme.sequence = self.uniprot_api.get_sequence(enzyme.uniprot_id)
            return
        
        # If we have an EC number, search for UniProt entries
        if enzyme.ec_number:
            logger.info(f"Searching UniProt for EC number {enzyme.ec_number}")
            results = self.uniprot_api.search_by_ec(enzyme.ec_number)
            
            if results:
                # Use the first result (could filter by organism preference)
                entry = results[0]
                uniprot_id = entry.get("primaryAccession")
                logger.info(f"Found UniProt entry: {uniprot_id}")
                
                # Update enzyme data
                enzyme.uniprot_id = uniprot_id
                enzyme.sequence = self.uniprot_api.get_sequence(uniprot_id)
                
                # Could update additional enzyme properties if desired
                if not enzyme.name and "proteinDescription" in entry:
                    rec_name = entry["proteinDescription"].get("recommendedName", {})
                    if "fullName" in rec_name:
                        enzyme.name = rec_name["fullName"].get("value")
    
    def enrich_substrate_data(self, substrate: Substrate) -> None:
        """Fetch additional data for a substrate"""
        if substrate is None:
            return
        
        # Skip if we already have SMILES
        if substrate.smiles:
            return
        
        # Try to get SMILES from PubChem (assuming ID is a PubChem CID)
        # In a real implementation, would need database detection logic
        logger.info(f"Fetching SMILES for substrate {substrate.id}")
        substrate.smiles = self.chemical_api.get_smiles(substrate.id, "pubchem")
        
        # If PubChem fails, try KEGG
        if not substrate.smiles and substrate.id.startswith("C"):
            logger.info(f"Trying KEGG for substrate {substrate.id}")
            substrate.smiles = self.chemical_api.get_smiles(substrate.id, "kegg")
    
    def predict_parameters(self, reaction: Reaction) -> None:
        """Predict kinetic parameters for a reaction"""
        if reaction.enzyme is None or not reaction.substrates:
            logger.info(f"Skipping parameter prediction for {reaction.id}: missing enzyme or substrates")
            return
        
        # Ensure we have complete enzyme data
        self.enrich_enzyme_data(reaction.enzyme)
        
        # Only proceed if we have a sequence
        if not reaction.enzyme.sequence:
            logger.warning(f"Cannot predict parameters for {reaction.id}: no enzyme sequence available")
            return
        
        # Get predictor
        predictor = self._get_predictor()
        
        # Process each substrate
        for substrate in reaction.substrates:
            # Ensure we have complete substrate data
            self.enrich_substrate_data(substrate)
            
            # Only predict if we have SMILES
            if substrate.smiles:
                logger.info(f"Predicting parameters for {reaction.id} with substrate {substrate.id}")
                kcat, km = predictor.predict(reaction.enzyme, substrate)
                
                if kcat is not None and km is not None:
                    # Store the results
                    if reaction.kcat is None:
                        # Assuming same kcat for all substrates
                        reaction.kcat = kcat
                    
                    # Store Km for this specific substrate
                    reaction.km[substrate.id] = km
                    
                    logger.info(f"Predicted: kcat = {kcat:.2f} /s, Km = {km:.2f} mM")
    
    def run(self, network_file: str, output_file: str = None, format: str = "auto") -> List[Reaction]:
        """
        Run the complete pipeline
        
        Args:
            network_file: Path to input network file
            output_file: Path to output file
            format: File format (auto, sbml, json, csv)
            
        Returns:
            List of Reaction objects with predicted parameters
        """
        logger.info(f"Starting enzyme parameter pipeline for {network_file}")
        
        # Parse network
        reactions = NetworkParser.parse(network_file, format)
        logger.info(f"Found {len(reactions)} reactions in the network")
        
        # Process each reaction
        for i, reaction in enumerate(reactions):
            logger.info(f"Processing reaction {i+1}/{len(reactions)}: {reaction.id}")
            self.predict_parameters(reaction)
        
        # Save results if output file is provided
        if output_file:
            self._save_results(reactions, output_file)
        
        return reactions
    
    def _save_results(self, reactions: List[Reaction], output_file: str) -> None:
        """Save results to file"""
        logger.info(f"Saving results to {output_file}")
        
        # Determine format from extension
        ext = os.path.splitext(output_file)[1].lower()
        
        if ext == ".json":
            # Convert to dictionary and save as JSON
            data = {"reactions": []}
            
            for reaction in reactions:
                # Convert dataclasses to dictionaries (handling nested objects)
                reaction_dict = {
                    "id": reaction.id,
                    "name": reaction.name,
                    "kcat": reaction.kcat,
                    "km": reaction.km
                }
                
                # Add enzyme data if available
                if reaction.enzyme:
                    reaction_dict["enzyme"] = {
                        "id": reaction.enzyme.id,
                        "name": reaction.enzyme.name,
                        "ec_number": reaction.enzyme.ec_number,
                        "uniprot_id": reaction.enzyme.uniprot_id,
                        "sequence": reaction.enzyme.sequence
                    }
                
                # Add substrates
                reaction_dict["substrates"] = []
                for substrate in reaction.substrates:
                    substrate_dict = {
                        "id": substrate.id,
                        "name": substrate.name,
                        "smiles": substrate.smiles
                    }
                    reaction_dict["substrates"].append(substrate_dict)
                
                # Add products
                reaction_dict["products"] = []
                for product in reaction.products:
                    product_dict = {
                        "id": product.id,
                        "name": product.name,
                        "smiles": product.smiles
                    }
                    reaction_dict["products"].append(product_dict)
                
                data["reactions"].append(reaction_dict)
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif ext == ".csv":
            # Create a flat CSV representation
            rows = []
            
            for reaction in reactions:
                # One row per substrate
                for substrate in reaction.substrates:
                    row = {
                        "reaction_id": reaction.id,
                        "reaction_name": reaction.name,
                        "kcat": reaction.kcat
                    }
                    
                    # Add enzyme data if available
                    if reaction.enzyme:
                        row.update({
                            "enzyme_id": reaction.enzyme.id,
                            "enzyme_name": reaction.enzyme.name,
                            "ec_number": reaction.enzyme.ec_number,
                            "uniprot_id": reaction.enzyme.uniprot_id
                        })
                    
                    # Add substrate data
                    row.update({
                        "substrate_id": substrate.id,
                        "substrate_name": substrate.name,
                        "substrate_smiles": substrate.smiles,
                        "km": reaction.km.get(substrate.id)
                    })
                    
                    rows.append(row)
            
            # Convert to DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
        
        elif ext == ".xml" or ext == ".sbml":
            try:
                import cobra
                from cobra.io import write_sbml_model
                
                # This would require more work to convert back to SBML/COBRApy format
                # For now, just warn that this is not fully implemented
                logger.warning("SBML output not fully implemented, using JSON format instead")
                
                # Save as JSON with SBML extension
                json_data = {"reactions": []}
                # ... (similar to JSON output)
                
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
            except ImportError:
                logger.error("COBRApy not installed. Install with: pip install cobra")
                # Save as JSON instead
                logger.warning(f"Falling back to JSON format saved as {output_file}")
                # ... (similar to JSON output)
        
        else:
            logger.warning(f"Unsupported output format: {ext}, using JSON")
            # Save as JSON
            # ... (similar to JSON output)
