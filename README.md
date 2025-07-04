# Quadratic Convolutional Tsetlin Machine (QC-TM)

**A novel interpretable machine learning architecture for natural language processing that combines the transparency of Tsetlin Machines with quadratic attention mechanisms inspired by transformers.**

## 🎓 About This Research

This repository contains the implementation of the **Quadratic Convolutional Tsetlin Machine (QC-TM)**, developed as part of a bachelor thesis exploring sustainable and interpretable alternatives to large language models in NLP.

### 🧠 The Problem

As large language models grow, they face critical challenges:
- **Decreased interpretability** - Black box decision making
- **High computational requirements** - Massive energy consumption  
- **Environmental impact** - Significant carbon footprint

Traditional Tsetlin Machines offer transparency and energy efficiency but suffer from limitations when processing natural language due to simple Set-of-Words (SoW) representations that lack contextual and positional information.

### 💡 Our Solution: QC-TM

The **Quadratic Convolutional Tsetlin Machine** extends the Convolutional Tsetlin Machine (CTM) by:

- ✅ **Maintaining interpretability** of traditional Tsetlin Machines
- ✅ **Incorporating quadratic word pair features** inspired by transformer attention
- ✅ **Adding positional awareness** through multi-hot encoding centered around each token
- ✅ **Filtering pairs using PMI/Cosine Similarity** for semantic relevance
- ✅ **Achieving competitive performance** while remaining energy-efficient

## 🏆 Performance Highlights

On text classification benchmarks, QC-TM demonstrates:
- **Outperforms RST model** by **30.97%** on FakeNewsNet dataset
- **Competitive with BERT** (only **0.93%** behind) while being more interpretable
- **Superior to traditional TM and CTM** across multiple datasets

## 🔬 Technical Innovation

### Multi-Hot Encoding Architecture
Unlike traditional TMs using Set-of-Words, QC-TM uses **five multi-hot encodings** around each token:

```
Position: [Before] [Previous] [Current] [Next] [After]
Channel:     0        1         2       3      4
```

### Quadratic Pair Features
- Creates **quadratic word pairs** similar to transformer attention
- Filters pairs using **Pointwise Mutual Information (PMI)** and **Cosine Similarity**
- Assigns pairs to positional channels based on **relative positioning**

### Interpretable Clauses
QC-TM generates human-readable clauses showing how it allocates features into the five multi-hot encodings for classification decisions.

## 🚀 Quick Start

### Prerequisites
```bash
pip install transformers datasets scipy scikit-learn pyyaml numpy
pip install PySparseCoalescedTsetlinMachineCUDA
```

### Run Experiments
```bash
# Clone repository
git clone <repo-url>
cd qc-tm

# Run with default config (CR dataset)
python main.py

# Or modify config.yaml for different experiments
```

## ⚙️ Configuration

Edit `config.yaml` to reproduce thesis experiments:

```yaml
qc-tm:
  preprocessing:
    dataset_name: "cr"          # cr, subj, pc
    maxlen: 20                  # Sequence length
    cosine_threshold: 0.7       # Cosine similarity filtering
    pmi_threshold: 0.5          # PMI filtering threshold
    num_words: 10000           # Vocabulary size
  
  training:
    epochs: 10                 # Training epochs
    s: 1.0                     # TM specificity parameter
    T: 10000                   # TM threshold parameter
    clauses: 10000             # Number of clauses
```

## 📊 Supported Datasets

| Dataset | Task | Performance vs BERT |
|---------|------|-------------------|
| **CR** | Sentiment Analysis | Competitive |
| **SUBJ** | Subjectivity Detection | Competitive |
| **PC** | Pro/Con Classification | Competitive |
| **FakeNewsNet** | Fake News Detection | -0.93% |

## 📁 Project Structure

```
├── qctm/
│   └── qc_tm.py              # QC-TM model implementation
├── data/
│   └── dataset.py            # Dataset loading utilities
├── preprocessing/
│   └── pre_processing.py     # Text preprocessing pipeline
├── thresholds/
│   ├── cosine.py            # Cosine similarity calculations
│   └── pmi.py               # PMI threshold calculations
├── config_loader.py          # Configuration management
├── config.yaml              # Experiment parameters
├── main.py                  # Main execution script
└── word_profile.p           # Pre-computed word embeddings
```

## 🔬 Research Methodology

### 1. Data Preprocessing
- **BERT Tokenization** for consistent vocabulary
- **Quadratic pair generation** between all words in sequence
- **PMI/Cosine filtering** to retain semantically meaningful pairs

### 2. Multi-Hot Encoding
Each token position gets 5 channels:
- **Channel 0**: Cosine similar words appearing before current token
- **Channel 1**: Previous word (n-gram context)
- **Channel 2**: Current word
- **Channel 3**: Next word (n-gram context)  
- **Channel 4**: Cosine similar words appearing after current token

### 3. Training & Evaluation
- **Sparse matrix representation** for memory efficiency
- **GPU acceleration** via CUDA implementation
- **Interpretable clause extraction** for model analysis

## 📈 Key Findings

1. **Competitive Performance**: QC-TM achieves near-BERT performance while maintaining interpretability
2. **Positional Awareness**: Multi-hot encoding successfully captures word positioning
3. **Effective Filtering**: PMI and cosine similarity filtering improves semantic understanding
4. **Interpretable Decisions**: Generated clauses reveal classification reasoning

## 🔮 Future Research Directions

Based on thesis conclusions, promising areas include:

- [ ] **Cosine similarity implementation** for improved pair filtering
- [ ] **BERT Tokenizer integration** for PMI calculations  
- [ ] **Bi-gram approaches** to complement current tri-gram functionality
- [ ] **Computational efficiency** improvements for larger datasets
- [ ] **Hyperparameter optimization** beyond current grid search

## 📚 Academic Context

This work contributes to the growing field of **interpretable AI** and **sustainable NLP**, offering a viable alternative to resource-intensive transformer models while maintaining competitive performance.


## 🤝 Contributing

This research implementation welcomes contributions, especially in areas identified for future work:

1. Fork the repository
2. Create a feature branch for your research contribution
3. Implement and test your changes
4. Submit a pull request with detailed methodology

## 📞 Contact & Collaboration

For academic collaboration, questions about the methodology, or access to additional experimental data:

- **Research Inquiries**: [thomasjh@uia.no]
- **Technical Issues**: Open a GitHub issue
- **Collaboration**: [University of Agder]

## 🏛️ License

This research code is available under the MIT License for academic and research purposes.

---

**🌟 Star this repository** if you're interested in interpretable AI and sustainable NLP research!

*"Bridging the gap between interpretability and performance in natural language processing"*
