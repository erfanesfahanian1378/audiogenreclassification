# Music Genre Classification using Machine Learning

**Course:** Audio Pattern Recognition  
**Institution:** Politecnico di Milano  
**Student:** Erfan Esfahanian  
**Supervisor:** Professor Stavros Ntalampiras  
**Date:** December 2024

---

## 📋 Project Overview

### Goal
Develop an automated music genre classification system that can accurately identify the genre of audio tracks using machine learning algorithms. The system analyzes acoustic features extracted from audio signals to classify songs into 10 different genres.

### Problem Statement
Music streaming platforms process millions of songs and need automated systems to categorize music by genre. Manual classification is time-consuming and subjective. This project demonstrates how machine learning can automate genre classification by analyzing audio features like timbre, rhythm, and spectral characteristics.

### Objectives
1. Extract meaningful audio features from raw audio signals
2. Apply unsupervised learning (k-means clustering) to visualize genre separation
3. Train and compare multiple supervised learning algorithms
4. Achieve competitive classification accuracy (target: >75%)
5. Provide model explainability through feature importance analysis
6. Identify which genres are most distinguishable and which are commonly confused

---

## 📊 Dataset

**GTZAN Genre Collection**
- **Source:** http://marsyas.info/downloads/datasets.html
- **Size:** 1,000 audio tracks
- **Duration:** 30 seconds per track
- **Genres:** 10 classes (100 songs each)
  - Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Format:** WAV files (22050 Hz sample rate)
- **Features:** Pre-extracted 58 audio features + manual extraction for validation

**Why GTZAN?**
- Standard benchmark dataset in Music Information Retrieval (MIR)
- Balanced dataset (equal samples per genre)
- Well-established baseline results for comparison
- Widely used in academic research

---

## 🗂️ Project Structure
```
audio_pattern_project/
├── Data/
│   ├── features_30_sec.csv          # Pre-extracted features (30s clips)
│   ├── features_3_sec.csv           # Pre-extracted features (3s segments)
│   ├── genres_original/             # Raw audio files (1000 WAV files)
│   └── images_original/             # Pre-generated spectrograms
├── notebooks/
│   ├── 01_data_exploration.ipynb           # Data loading and quality checks
│   ├── 02_audio_visualization.ipynb        # Audio waveforms and spectrograms
│   ├── 03_feature_extraction.ipynb         # Manual feature extraction
│   ├── 04_baseline_models.ipynb            # ML model training and evaluation
│   └── 05_kmeans_clustering.ipynb          # Unsupervised clustering analysis
├── results/
│   └── figures/                     # All generated visualizations
├── models/                          # Saved trained models (if any)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 📓 Detailed Notebook Descriptions

### **01_data_exploration.ipynb** - Data Loading and Quality Assessment

**Purpose:** Understand the dataset structure, verify data quality, and establish baseline understanding.

**What We Did:**
1. **Loaded Pre-extracted Features**
   - Read CSV files containing 58 audio features for each song
   - Features include MFCCs, spectral features, chroma, tempo, etc.
   
2. **Data Quality Checks**
   - Verified no missing values (all 1000 samples complete)
   - Confirmed no duplicate entries
   - Checked data types (numeric features as float64)
   
3. **Genre Distribution Analysis**
   - Confirmed balanced dataset: exactly 100 songs per genre
   - Created bar chart visualization of genre distribution
   
4. **Feature Statistics**
   - Calculated mean, std, min, max for all features
   - Identified feature value ranges (e.g., MFCCs: -552 to +8263268)
   - Noted that features have different scales → normalization needed
   
5. **Audio File Verification**
   - Verified all 1000 WAV files exist in genres_original folder
   - Confirmed 10 genre subdirectories with 100 files each

**Key Findings:**
- Dataset is clean and well-structured
- Perfectly balanced (no class imbalance issues)
- Features have vastly different scales (requires normalization)
- Ready for machine learning pipeline

**Outputs:**
- `01_genre_distribution.png` - Bar chart of genre counts

---

### **02_audio_visualization.ipynb** - Audio Signal Analysis

**Purpose:** Visualize audio signals in time and frequency domains to understand acoustic differences between genres.

**What We Did:**
1. **Loaded Raw Audio Files**
   - Used librosa to load WAV files (22050 Hz sampling rate)
   - Selected representative samples from each genre
   
2. **Waveform Visualization**
   - Plotted amplitude over time for sample songs
   - Observed different envelope patterns across genres
   - **Blues:** Consistent amplitude, steady rhythm
   - **Classical:** Dynamic range (quiet → loud transitions)
   - **Metal:** High amplitude throughout, aggressive
   - **Jazz:** Medium amplitude with rhythmic variation
   
3. **Spectrogram Generation**
   - Created time-frequency representations using STFT
   - Converted to decibel scale for better visualization
   - **Key observations:**
     - Metal: Dense across all frequencies (distorted guitars)
     - Classical: Clear harmonic structure (orchestral)
     - Jazz: Complex mid-range patterns (saxophone, bass)
   
4. **Multi-Genre Comparison**
   - Created side-by-side waveform comparisons
   - Clearly showed visual differences in audio signatures
   - Validated that features should capture these differences

**Key Insights:**
- Different genres have visually distinct waveform patterns
- Spectrograms reveal frequency content differences
- These visual differences justify using audio features for ML
- Human-visible patterns → ML-learnable patterns

**Outputs:**
- `02_waveform_blues.png` - Example waveform
- `02_spectrogram_blues.png` - Example spectrogram
- `02_waveform_comparison.png` - Multi-genre comparison

---

### **03_feature_extraction.ipynb** - Manual Feature Extraction

**Purpose:** Extract audio features manually from raw audio to understand what each feature represents and validate pre-extracted features.

**What We Did:**
1. **Extracted Core Audio Features**
   
   **A. MFCCs (Mel-Frequency Cepstral Coefficients) - 40 features**
   - 20 MFCC coefficients × 2 statistics (mean, variance)
   - Captures timbre/texture - the "fingerprint" of sound
   - Most important features for genre classification
   - Represents how humans perceive sound spectrally
   
   **B. Chroma Features - 2 features**
   - Mean and variance of 12 pitch classes
   - Represents musical notes (C, C#, D, ..., B)
   - Useful for identifying harmonic content
   - Classical music has clear chroma patterns
   
   **C. Spectral Features - 6 features**
   - **Spectral Centroid:** "Brightness" (where frequencies concentrate)
     - High → Bright sound (cymbals, high vocals)
     - Low → Dark sound (bass, low instruments)
   - **Spectral Rolloff:** Frequency below which 85% of energy lies
   - **Spectral Bandwidth:** Spread of frequencies
   
   **D. Temporal Features - 3 features**
   - **Zero Crossing Rate (ZCR):** Noisiness/percussion measure
     - High ZCR → Noisy (metal, drums)
     - Low ZCR → Smooth (strings, vocals)
   - **Tempo:** Speed in beats per minute (BPM)
   
   **E. Energy Features - 2 features**
   - **RMS (Root Mean Square):** Overall loudness/energy
   - Metal has high RMS, Classical varies

2. **Visualized MFCC Patterns**
   - Created heatmaps showing MFCCs over time
   - Different genres have distinct MFCC "signatures"

3. **Compared Features Across Genres**
   - Created boxplots comparing feature distributions
   - **Key findings:**
     - Metal: Highest spectral centroid (brightest)
     - Jazz: Lowest spectral centroid (darkest)
     - Metal: Highest zero crossing rate (noisiest)
     - Classical vs Metal: Most different in all features

**Key Insights:**
- MFCCs are most information-rich features
- Spectral centroid clearly separates Metal from other genres
- Zero crossing rate distinguishes noisy vs smooth genres
- Feature extraction converts audio → numbers for ML
- Pre-extracted features validated by manual extraction

**Outputs:**
- `03_mfcc_visualization.png` - MFCC heatmap
- `03_feature_comparison.png` - Feature boxplots across genres

---

### **04_baseline_models.ipynb** - Machine Learning Model Training

**Purpose:** Train multiple classification algorithms, evaluate performance, and identify the best model for genre classification.

**What We Did:**

1. **Data Preparation**
   - Loaded pre-extracted features (1000 songs × 58 features)
   - Separated features (X) from labels (y)
   - **Train-test split:** 80% training (800 songs), 20% testing (200 songs)
   - **Stratified split:** Maintains genre distribution in both sets
   - **Feature normalization:** StandardScaler (mean=0, std=1)
     - Critical for distance-based algorithms
     - Ensures all features contribute equally

2. **Model 1: Random Forest Classifier**
   - **Algorithm:** Ensemble of 100 decision trees
   - **How it works:** Each tree votes, majority wins
   - **Parameters:** 100 estimators, unlimited depth
   - **Training time:** Fast (~2 seconds)
   - **Results:**
     - **Accuracy: 77.5%** (155/200 correct)
     - **Best performing model**
     - Good balance between bias and variance
     - Handles high-dimensional data well

3. **Model 2: Support Vector Machine (SVM)**
   - **Algorithm:** Finds optimal hyperplane to separate classes
   - **Kernel:** RBF (Radial Basis Function) for non-linear boundaries
   - **Parameters:** C=10, gamma='scale'
   - **Results:**
     - **Accuracy: 76.5%** (153/200 correct)
     - Very close to Random Forest
     - Good for high-dimensional feature spaces
     - Slightly slower training

4. **Model 3: K-Nearest Neighbors (KNN)**
   - **Algorithm:** Classifies based on 5 nearest neighbors
   - **Distance metric:** Euclidean distance
   - **No training:** Simply memorizes training data
   - **Results:**
     - **Accuracy: 70.0%** (140/200 correct)
     - Simplest algorithm but lowest accuracy
     - Sensitive to feature scaling (we normalized)
     - Slower prediction time

5. **Model Comparison**
   - Created bar chart comparing accuracies
   - **Winner: Random Forest (77.5%)**
   - All models performed reasonably well (>70%)

6. **Detailed Performance Analysis (Random Forest)**
   
   **Classification Report Insights:**
   - **Best classified genres:**
     - Classical: 95% (19/20 correct)
     - Pop: 90% (18/20 correct)
     - Metal: 85% (17/20 correct)
     - Jazz: 85% (17/20 correct)
   
   - **Hardest genres:**
     - Rock: 55% (11/20 correct)
     - Disco: 60% (12/20 correct)
     - Country: 75% (15/20 correct)
   
   - **Why differences?**
     - Classical has unique orchestral features
     - Rock is very diverse (soft rock ≈ blues, hard rock ≈ metal)
     - Disco confused with Hip-hop (both electronic, danceable)

7. **Confusion Matrix Analysis**
   - Created heatmap showing misclassifications
   - **Key confusions:**
     - Disco ↔ Hip-hop (5 songs): Similar electronic beats
     - Rock ↔ Country (3 songs): Folk rock overlap
     - Blues ↔ Jazz (2 songs): Similar instruments/style
     - Rock ↔ Metal (1 song): Related genres
   - **Insight:** Mistakes make musical sense!

8. **Feature Importance Analysis (Explainability)**
   - Extracted importance scores from Random Forest
   - **Top 5 most important features:**
     1. chroma_stft_mean (4.4%)
     2. perceptr_var (3.8%)
     3. length (3.7%)
     4. chroma_stft_var (3.1%)
     5. rms_var (3.1%)
   
   - **Feature type breakdown (top 20):**
     - MFCCs: ~40% of important features
     - Spectral: ~30%
     - Chroma: ~15%
     - Others: ~15%
   
   - **Key insight:** No single feature dominates; ensemble of many features works best
   - **Model explainability:** We understand which acoustic properties matter

**Key Findings:**
- Random Forest achieves best performance (77.5%)
- Results comparable to published research on GTZAN
- Some genres naturally overlap (explains why not 100%)
- Model decisions are explainable through feature importance
- Confusion patterns match acoustic similarity

**Outputs:**
- `04_model_comparison.png` - Bar chart of model accuracies
- `04_confusion_matrix.png` - Heatmap showing misclassifications
- `04_feature_importance.png` - Bar chart of top 20 features

---

### **05_kmeans_clustering.ipynb** - Unsupervised Learning Analysis

**Purpose:** Use k-means clustering to validate that features naturally separate genres, providing independent verification that our feature extraction works.

**What We Did:**

1. **Applied K-Means Clustering**
   - Algorithm: Unsupervised clustering (no genre labels given)
   - Number of clusters: k=10 (matching 10 genres)
   - Distance metric: Euclidean distance in 58D feature space
   - **Process:** Algorithm finds 10 cluster centers that minimize within-cluster variance

2. **Clustering Quality Metrics**
   
   **A. Silhouette Score: 0.107**
   - Measures: How well-separated clusters are
   - Scale: [-1, 1], higher is better
   - **Interpretation:** Low score indicates overlapping clusters
   - **Why low?** Music genres naturally overlap in feature space
   - **Normal for this problem:** Genre boundaries are fuzzy
   
   **B. Adjusted Rand Index (ARI): 0.182**
   - Measures: Agreement between clusters and true genres
   - Scale: [-1, 1], higher is better
   - **Interpretation:** Clusters partially match genres
   - **Insight:** Some genres cluster well, others mixed
   
   **C. Normalized Mutual Information (NMI): 0.326**
   - Measures: Information shared between clusters and genres
   - Scale: [0, 1], higher is better
   - **Interpretation:** Knowing cluster gives moderate info about genre
   - **Conclusion:** Features capture some genre structure

3. **Cluster Composition Analysis**
   - Created crosstab: cluster ID vs genre labels
   - **Pure clusters (>50% one genre):**
     - Cluster 5: 63.4% Classical
     - Cluster 1: 55.7% Blues
     - Cluster 4: 53.7% Pop
   
   - **Mixed clusters:**
     - Cluster 2: 18% Country, 17% Rock, 16% Blues (very mixed)
     - Cluster 3: 25% Disco, 22% Pop, 17% Rock
   
   - **Insight:** Classical, Blues, and Pop are most distinct
   - **Challenge:** Country, Rock, Disco form mixed clusters

4. **Dimensionality Reduction with PCA**
   - **Problem:** Can't visualize 58-dimensional data
   - **Solution:** Principal Component Analysis (PCA)
   - Reduced 58 dimensions → 2 dimensions (PC1, PC2)
   - **Explained variance: 40.22%**
     - PC1: 23.57% of total variance
     - PC2: 16.65% of total variance
   - **Interpretation:** 2 dimensions capture 40% of information (reasonable)

5. **PCA Visualization**
   - Created two side-by-side scatter plots
   
   **Left plot: True genre colors**
   - Shows natural genre distribution in feature space
   - **Observations:**
     - Metal cluster is separated (bottom left)
     - Classical spread out (some overlap with Jazz)
     - Blues/Country/Rock form large overlapping region
     - Pop/Hip-hop in upper right region
   
   **Right plot: K-means cluster colors**
   - Shows how algorithm grouped songs
   - Red X markers = cluster centers
   - **Observations:**
     - Clusters roughly follow genre patterns
     - Some clusters pure, some mixed
     - Validates feature extraction approach

6. **Genre Separation Analysis**
   - Calculated average PCA position for each genre
   - Computed pairwise distances between genre centers
   
   **Most similar genres (small distance):**
   - Blues ↔ Country: 0.95 (closest!)
   - Blues ↔ Jazz: 1.12
   - Country ↔ Jazz: 1.47
   - Disco ↔ Rock: 1.50
   
   **Most different genres (large distance):**
   - Metal ↔ Pop: 9.60 (most different!)
   - Classical ↔ Metal: 9.22
   - Classical ↔ Pop: 8.87
   - Metal ↔ Reggae: 7.86
   
   **Insight:** Explains Random Forest confusion patterns!

7. **Cluster Purity Heatmap**
   - Visualized: What % of each cluster is each genre
   - **Ideal:** One dark red cell per row (pure cluster)
   - **Reality:** Some mixed clusters (normal)
   - **Best clusters:** Classical, Pop, Blues
   - **Most mixed:** Country/Rock/Blues cluster

**Key Findings:**
- Unsupervised learning confirms feature quality
- Some genres naturally cluster well (Classical, Metal, Pop)
- Some genres overlap significantly (Blues/Country/Jazz, Rock/Metal)
- Clustering patterns match supervised learning errors
- Feature extraction successfully captures genre characteristics
- 40% variance in 2D is reasonable for complex music data

**Outputs:**
- `05_kmeans_pca_visualization.png` - Side-by-side PCA plots
- `05_cluster_purity_heatmap.png` - Cluster composition heatmap

---

## 🎯 Key Results Summary

### Model Performance
- **Best Model:** Random Forest
- **Accuracy:** 77.5% (on 200 unseen test songs)
- **Comparison:** Comparable to published research on GTZAN (70-85% typical)

### Genre-Specific Performance
| Genre | Accuracy | Notes |
|-------|----------|-------|
| Classical | 95% | Most distinctive (orchestral features) |
| Pop | 90% | Clear modern production |
| Metal | 85% | Bright, noisy, high energy |
| Jazz | 85% | Dark, smooth, complex |
| Blues | 75% | Overlaps with Jazz |
| Country | 75% | Overlaps with Blues/Rock |
| Hip-hop | 75% | Sometimes confused with Disco |
| Reggae | 80% | Distinctive rhythm |
| Disco | 60% | Often confused with Hip-hop |
| Rock | 55% | Very diverse genre (hardest to classify) |

### Feature Importance
- **Most important:** Chroma features (pitch content)
- **Strong contributors:** MFCCs (timbre), Spectral features (brightness)
- **Moderate contributors:** Energy features (loudness)
- **Less important:** Tempo (not discriminative enough)

### Clustering Insights
- Classical, Pop, Metal form distinct clusters
- Blues, Jazz, Country share feature space
- Rock is most diverse (scattered across multiple clusters)
- Disco and Hip-hop overlap significantly

---

## 💡 Key Learnings

1. **Feature Engineering is Critical**
   - Audio features successfully capture genre characteristics
   - Multiple feature types needed (no single feature sufficient)
   - MFCCs most important overall

2. **Genre Boundaries are Fuzzy**
   - Some genres naturally sound similar
   - 100% accuracy unrealistic (even humans disagree)
   - Confusion patterns make musical sense

3. **Model Selection Matters**
   - Ensemble methods (Random Forest) perform best
   - Simple models (KNN) struggle with high-dimensional data
   - SVM competitive but slower

4. **Explainability is Achievable**
   - Feature importance reveals model reasoning
   - Not a "black box" - decisions are interpretable
   - Clustering validates supervised learning results

5. **Dataset Balance Helps**
   - Equal samples per genre prevents bias
   - Stratified splitting maintains distribution
   - Fair evaluation across all genres

---

## 🔧 Requirements
```
Python 3.8+
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
librosa>=0.10.1
scikit-learn>=1.3.0
jupyter>=1.0.0
ipykernel>=6.25.0
soundfile>=0.12.1
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Clone/Download Project**
```bash
   cd audio_pattern_project
```

2. **Install Dependencies**
```bash
   pip install -r requirements.txt
```

3. **Launch Jupyter**
```bash
   jupyter notebook
```

4. **Run Notebooks in Order**
   - Start with `01_data_exploration.ipynb`
   - Run cells sequentially (Shift + Enter)
   - Each notebook builds on previous ones
   - All visualizations save to `results/figures/`

5. **View Results**
   - Check `results/figures/` for all plots
   - Review confusion matrix for error analysis
   - Examine feature importance for explainability

---

## 🧠 Theoretical Concepts

### 1. Mel-frequency Cepstral Coefficients (MFCCs)
**What they are:** MFCCs are coefficients that collectively make up an MFC (Mel-frequency cepstrum). They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum").
**Why we use them:** They approximate the human auditory system's response more closely than linearly-spaced frequency bands. In music classification, MFCCs are the primary feature for capturing "timbre" or "tone color"—the quality that makes a guitar sound different from a piano even when playing the same note.

### 2. Random Forest Classifier
**What it is:** An ensemble learning method that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees (the mode).
**Why we use it:** 
- **Robustness:** It corrects for decision trees' habit of overfitting to their training set.
- **Feature Importance:** It provides a clear metric for which features (like specific MFCCs or spectral centroid) are most critical for distinguishing genres.
- **High Dimensionality:** It handles our 58-feature dataset effectively without requiring extensive feature selection beforehand.

### 3. K-Means Clustering
**What it is:** An unsupervised learning algorithm that partitions $n$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean (cluster center).
**Why we use it:** Since K-means doesn't use the genre labels, we use it to validate our features. If the algorithm naturally groups "Classical" songs together and "Metal" songs together based solely on their audio features, it proves that our extracted features are meaningful and robust.

### 4. Principal Component Analysis (PCA)
**What it is:** A dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information in the large set. It creates new uncorrelated variables (Principal Components) that maximize variance.
**Why we use it:** We cannot visualize a 58-dimensional space. PCA allows us to project this complex data onto a 2D plane (PC1 and PC2) while retaining ~40% of the original information, enabling us to visually inspect how genres overlap or separate in the feature space.

---

## 📚 References

1. **Dataset:**
   - Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302.

2. **Libraries:**
   - Librosa: McFee, B., et al. (2015). librosa: Audio and music signal analysis in python.
   - Scikit-learn: Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.

3. **Related Work:**
   - Fu, Z., et al. (2011). A survey of audio-based music classification and annotation.
   - Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use.

---

## 🎓 Author

**Erfan Esfahanian**  
Student at Politecnico di Milano  
Course: Audio Pattern Recognition  
Supervisor: Professor Stavros Ntalampiras  
Date: December 2024

---

## 📄 License

This project is for academic purposes only.  
Dataset: GTZAN is publicly available for research use.

---

## 🙏 Acknowledgments

- Professor Stavros Ntalampiras for guidance
- GTZAN dataset creators
- Librosa and scikit-learn communities
- Audio Pattern Recognition course materials

---

**For questions or discussion about this project, contact the author or professor.**