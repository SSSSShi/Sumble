# Team Name Normalization Project

This project normalizes team names extracted from job postings to identify unique teams within companies.

## Project Structure

```
.
├── teams.csv                 # Input data
├── output.csv                # Generated output with normalized teams
├── team_normalizer.py        # Main normalization pipeline
├── validate_output.py        # Quality analysis and validation
├── explore_data.py           # Data exploration script
├── approach.md               # Summary of approach
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install pandas scikit-learn rapidfuzz
```

### Running the Pipeline

```bash
# Generate normalized output
python3 team_normalizer.py

# Validate results
python3 validate_output.py

# Explore data (optional)
python3 explore_data.py
```

## Approach

The normalization pipeline uses a multi-stage approach:

1. **Text Normalization**: Lowercase conversion, punctuation removal, whitespace normalization
2. **Acronym Extraction**: Identifies patterns like "Technology Development Group (TDG)" and maps acronyms to full forms
3. **Suffix Removal**: Removes common organizational suffixes (team, group, organization, etc.)
4. **Fuzzy Matching**: Uses RapidFuzz with 85% similarity threshold to group similar team names
5. **Bidirectional Validation**: Ensures important keywords are preserved and canonical names make sense
6. **Company-Specific Processing**: Processes each company separately for better accuracy

## Results

- **Input**: 51,162 team mentions, 6,241 unique teams
- **Output**: 4,215 normalized teams (32.5% reduction)
- **High precision**: Bidirectional validation ensures semantic correctness
- **Preserves context**: Domain prefixes (RF, CPU, GPU) and important keywords (system, analysis, search) maintained
- **Quality assurance**: 9,338 potential false matches prevented by validation layer

### Example Normalizations

**Apple**: "Technology Development Group (TDG)" → "Technology Development"
- Technology Development Group
- Technology Development Group (TDG)
- technology development group

**OpenAI**: "Applied AI" variations → "Applied Ai"
- Applied AI team
- Applied AI Engineering team
- Applied AI

**Validation in action** (prevented false matches):
- ✗ "cellular system engineering" ≠ "Cellular Team" (preserves "system")
- ✗ "RF Platform Architecture" ≠ "CPU Platform Architecture" (different domains)
- ✗ "Active Safety Test" ≠ "Passive Safety Test" (opposing keywords)

## Design Tradeoffs & Future Improvements

The current implementation prioritizes precision over recall, ensuring semantic correctness through bidirectional validation. This means we preserve important context (e.g., "Search" in "Apple Media Products Search Team" → "Apple Media Products Search") at the cost of potentially missing some valid consolidations. The validation layer prevents 9,338+ false matches by checking that: (1) all words in canonical names appear in originals, and (2) important keywords like "system", "analysis", "search" are preserved from originals.

Future iterations could employ a hierarchical team structure to capture both parent organizations and sub-teams simultaneously, enabling multi-level grouping (e.g., "Apple Media Products" as parent with "Search" as child). The 85% fuzzy matching threshold could be tuned per-company based on validation data, and machine learning approaches (clustering, entity resolution models, word embeddings) could further improve accuracy. The modular architecture makes these enhancements straightforward to implement.

## Code Quality Features

- **Clean, modular object-oriented design**: Reusable `TeamNormalizer` class with single responsibility methods
- **Type hints**: Full type annotations for better code clarity and IDE support
- **Comprehensive docstrings**: Every method documented with clear explanations
- **Validation layer**: Bidirectional validation prevents 9,338 false matches
- **Rejection tracking**: Captures and analyzes why matches were rejected for quality insights
- **Analysis scripts**: validate_output.py provides detailed quality metrics and validation reports
- **Extensible architecture**: Easy to add new companies, adjust thresholds, or integrate into pipelines

## Usage as Library

```python
from team_normalizer import TeamNormalizer

# Initialize normalizer
normalizer = TeamNormalizer(fuzzy_threshold=85)

# Normalize a DataFrame
df = pd.read_csv('teams.csv')
result_df = normalizer.normalize_dataframe(df)

# Or normalize individual team names
normalized = normalizer.normalize_team_name("Engineering Team")
```