# Spotify Songs **Health Impact** Analysis

This repository contains a two–part project that combines Spotify audio features with the MxMH mental-health survey to label each track’s potential effect on a listener as **Calming · Neutral · Anxious**:

1. **Jupyter Notebook (`SpotifyRuhAnalizi.ipynb`)**  
   - Full data pipeline: cleaning, exploratory analysis, statistical testing, model training.
2. **Streamlit Web App (`uygulama.py`)**  
   - Re-implements the notebook workflow and serves an interactive demo where users can pick a song and instantly see its predicted health impact and class probabilities.

<br/>

## Project Structure


├── dataset.csv # Spotify audio features (≈160 k tracks)

├── mxmh_survey_results.csv # MxMH mental-health survey

├── SpotifyRuhAnalizi.ipynb #  Main notebook – analysis & modeling

├── uygulama.py #  Streamlit interface

├── requirements.txt # Python dependencies



> **Note:** The CSV files are provided for academic use only.

<br/>

## Jupyter Notebook – Analysis & Modeling


```bash
| Section | What it does | Output |
|---------|--------------|--------|
| **Intro & Literature** | Project goals, data sources, music-psychology background | Markdown |
| **Data Loading** | Import `dataset.csv` & `mxmh_survey_results.csv` | Pandas preview |
| **Pre-processing** | Handle missing values, map *Never / Rarely …* to numeric | Code + tables |
| **Genre-Level Scores** | Mean **Anxiety / Depression / Insomnia / OCD** for each genre | Heatmap |
| **K-Means (k = 3)** | Cluster genres into *Calming / Neutral / Anxious* | Scatter & Silhouette |
| **PCA** | Dimensionality reduction, variance explained | 2-D plot |
| **Hypothesis Tests** | Levene, ANOVA, Kruskal-Wallis | p-value tables |
| **GaussianNB Model** | 7 audio features → class label | Confusion matrix |
| **Conclusion** | Statistical insights, limitations, next steps | Markdown |
```

### Run the Notebook

```bash
git clone https://github.com/yourusername/spotify-health-effect.git
cd spotify-health-effect

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter lab SpotifyRuhAnalizi.ipynb
```

### Streamlit App – Live Demo
The Streamlit script retrains the same GaussianNB model and exposes it through a simple UI.

Select any song to see its audio features and the predicted class probabilities.

```bash
streamlit run uygulama.py
```
Then open http://localhost:8501 in your browser.


### Streamlit App Screenshot
![Ekran görüntüsü 2025-06-12 225646](https://github.com/user-attachments/assets/5287444f-2f5e-416e-8433-11f1d7d901eb)
