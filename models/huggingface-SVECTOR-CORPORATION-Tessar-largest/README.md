---
language: en
tags:
- tessar
- table-question-answering
- svector
- neural-sql-executor
datasets:
- Stanford/wikitablequestions
license: mit
pipeline_tag: table-question-answering
---

# Tessar (Large-Sized Model)

Tessar is an advanced table reasoning model developed by SVECTOR, building upon the groundbreaking research and pushing the boundaries of neural table understanding.

## Model Description

Tessar (**Te**xtual **S**QL **A**nalysis and **R**easoning) is a sophisticated neural model designed to excel in table-based question answering. Tessar implements an innovative neural SQL executor that can interpret and reason over complex tabular data with remarkable precision.

The model is constructed using the BART architecture, featuring a bidirectional encoder and an autoregressive decoder. This design allows Tessar to capture intricate contextual relationships within tabular data and generate accurate, contextually relevant answers.

### Key Features
- Advanced neural SQL execution capabilities
- State-of-the-art performance on complex table question answering
- Robust handling of nuanced and multi-step queries
- Fine-tuned on the WikiTableQuestions dataset

## Intended Uses

Tessar is particularly powerful for solving complex table-based questions across various domains. Here are some example questions the model can effectively address:

| Question | Example Answer |
|:---: |:---:|
| According to the table, what is the last title produced? | Specific Title |
| What is the difference in a specific comparative metric? | Numerical Difference |
| Which entity had the most significant impact in a given context? | Identified Entity |
| What were the first and last entries in a specific column? | Comparative Entries |

### How to Use

Here's a comprehensive example of using Tessar with the Transformers library:

```python
from transformers import TessarTokenizer, BartForConditionalGeneration
import pandas as pd

# Load Tessar model and tokenizer
tokenizer = TessarTokenizer.from_pretrained("SVECTOR-CORPORATION/Tessar-largest")
model = BartForConditionalGeneration.from_pretrained("SVECTOR-CORPORATION/Tessar-largest")

# Prepare sample table data
data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)

# Ask a specific query
query = "In which year did beijing host the Olympic Games?"
encoding = tokenizer(table=table, query=query, return_tensors="pt")

# Generate answer
outputs = model.generate(**encoding)

# Decode and print result
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# Expected output: [' 2008.0']
```

### Evaluation

For comprehensive evaluation scripts and benchmarking, please refer to the SVECTOR documentation and research repositories.

### Performance Highlights
- Exceptional accuracy on complex table reasoning tasks
- Robust handling of multi-step and contextual queries
- State-of-the-art performance on WikiTableQuestions dataset

### Citation
  
If you use Tessar in your research the SVECTOR implementation:

```bibtex
@inproceedings{svector2025tessar,
    title={{Tessar}: Advanced Neural Table Reasoning},
    author={{SVECTOR Team}},
    year={2025},
    institution={SVECTOR Research}
}
```

### Contact and Support

For further information, support, or collaboration opportunities, please contact SVECTOR's research team at research@svector.co.in.

### License

This model is released under the MIT License. Please review the licensing terms before use.