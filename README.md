# California Wildfire Damage Prediction

Machine learning analysis comparing CART, KNN, and Naive Bayes models to predict wildfire damage severity.

## Dataset
- **Source**: California Wildfire Data (55,750 records)
- **Features**: 24 features (10 numerical, 14 categorical)
- **Cleaned**: 29,430 records after removing invalid/missing data

## Results
| Model | Test R² | RMSE |
|-------|---------|------|
| **CART** | **0.8420** | **0.6281** |
| KNN | 0.8232 | 0.6645 |
| Naive Bayes | -0.0516 | 1.6205 |

**Winner**: CART with 84.2% variance explained and minimal overfitting.

**Agentic Model**: We also use Claude Haiku 4.5 and E2B as an agentic model to generate its own EDA, cleaning, and modeling code to arrive it its own results.
This model also found CART to be the winner, verifying our pick.

## Requirements
- Python 3.10+
- Anthropic API key (for Claude Haiku 4.5 model)
- E2B API key (for sandboxed code execution)

## Usage
```python
# Install dependencies
pip install pandas scikit-learn numpy matplotlib e2b-code-interpreter python-dotenv anthropic

# Set environment variables based on .env.example
# ANTHROPIC_API_KEY="your_key_here"
# E2B_API_KEY="your_key_here"

# Run analysis
jupyter notebook final.ipynb
```

## Key Findings
- CART achieves superior predictive accuracy with optimal generalization
- Decision tree model provides interpretable damage predictions
- Suitable for real-time damage assessment and resource allocation
