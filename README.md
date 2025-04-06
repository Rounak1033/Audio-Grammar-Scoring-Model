# 🗣️ Grammar Scoring Engine for Voice Samples

This project is developed for the **SHL Intern Hiring Assessment** on Kaggle. The task is to build a machine learning model that can accurately **score spoken English grammar quality** (on a scale from 0 to 5) based on `.wav` audio recordings.



## 📁 Dataset

- **Train Set**: 390 audio files + labels (0–5 grammar scores)
- **Test Set**: 195 audio files (to predict scores)
- Audio format: `.wav`, sampled at 16kHz
- Data provided through Kaggle competition interface



## 🔧 Approach

1. **Audio Preprocessing**
   - Used `librosa` to extract MFCC features from each `.wav` file
   - Sample rate fixed at 16,000 Hz
   - Extracted 40 MFCCs per sample

2. **Feature Engineering**
   - Took statistical aggregations (mean, std, min, max) of MFCCs
   - Normalized features using `StandardScaler`

3. **Modeling**
   - Used **LightGBM Regressor** for its speed and performance
   - Trained using 5-fold cross-validation
   - Final model trained on full dataset and used for inference

4. **Evaluation Metric**
   - Pearson Correlation between predicted and true scores



## 📊 Performance

- **Public Kaggle Score**: `0.693` (Pearson Correlation)



## 🧠 Dependencies

- Python 3.x
- `numpy`, `pandas`, `librosa`
- `scikit-learn`, `lightgbm`
- Compatible with Kaggle notebook environment



## 💡 Future Improvements

- Use spectrograms or pretrained audio models (e.g., wav2vec, Whisper)
- Ensemble multiple models for higher correlation
- Improve feature extraction with delta/acceleration MFCCs
