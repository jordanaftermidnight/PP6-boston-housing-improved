# 🏠 PP6: Boston Housing Price Prediction - Improved Model

A machine learning project that tackled real-world data challenges and delivered models that significantly outperformed our starting benchmarks through careful feature crafting and strategic optimization.

## 🎯 Project Overview

This project enhances the classic Boston Housing dataset analysis by implementing:
- **Feature Engineering**: 8 new engineered features based on domain knowledge
- **Advanced Preprocessing**: Outlier detection and removal using statistical methods
- **Model Optimization**: Improved neural network architecture with regularization
- **Comprehensive Validation**: Multi-run stability analysis and detailed performance metrics
- **Data Visualization**: Professional plots and analysis charts

## 📊 Key Results

- **MSE Improvement**: ~20% reduction in Mean Squared Error
- **R² Enhancement**: Significant improvement in R-squared scores
- **Model Stability**: Consistent performance across multiple runs
- **Feature Insights**: Clear visualization of feature importance and correlations

## 📝 Assignment Deliverables

### 📓 PP6 Assignment Notebook
- **GitHub URL**: https://github.com/jordanaftermidnight/PP6-boston-housing-improved/blob/main/PP6_Boston_Housing_Improvements.ipynb
- **Local Path**: `PP6_Boston_Housing_Improvements.ipynb`

### ✍️ Written Summary
Looking at the original Boston Housing regression problem, I decided to push beyond basic linear models and implement several key improvements that would make a real difference in prediction accuracy. The main areas I focused on were advanced feature engineering, neural network architecture optimization, and comprehensive validation techniques. I added 8 carefully crafted features like interaction terms (LSTAT×RM), polynomial features (RM²), and ratio features (PTRATIO/TAX) that capture non-linear relationships the original features couldn't express. The neural network architecture got a complete overhaul with batch normalization, dropout regularization, and early stopping callbacks that prevent overfitting while maintaining model capacity. I also implemented robust outlier detection using Isolation Forest, which cleaned up the data and improved model stability. The visualization suite now includes error vs epoch plots, residual analysis, and feature importance rankings that give real insights into model behavior. Most importantly, I added multi-run stability analysis to ensure the improvements weren't just lucky random variations but consistent performance gains. The end result was a 10-20% improvement in MSE with much better generalization characteristics.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup_project.sh
./setup_project.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p visualizations models results

# Run the project
python boston_housing_improved.py
```

### Option 3: Using Make
```bash
make all  # Setup and run everything
```

## 📁 Project Structure

```
PP6-boston-housing-improved/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup_project.sh                   # Automated setup script
├── Makefile                           # Make commands
├── .gitignore                         # Git ignore rules
├── boston_housing_improved.py         # Main analysis script
├── boston_housing_improved.ipynb      # Jupyter notebook version
├── check_setup.py                    # Environment verification
├── LAUNCH_COMMANDS.md                 # Detailed usage guide
├── venv/                             # Virtual environment
├── models/                           # Trained models
│   └── boston_housing_improved_model.h5
├── results/                          # Analysis results
│   ├── improvement_results.json
│   └── improvement_summary.txt
└── visualizations/                   # Generated plots
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── feature_importance.png
```

## 🔧 Features

### Data Preprocessing
- **Outlier Detection**: Statistical methods to identify and remove outliers
- **Feature Scaling**: StandardScaler for optimal model performance
- **Data Validation**: Comprehensive checks for data quality

### Feature Engineering
- **Interaction Features**: LSTAT × RM, CRIM × RAD interactions
- **Polynomial Features**: RM², LSTAT² for non-linear relationships  
- **Ratio Features**: PTRATIO/TAX, B/NOX ratios
- **Binned Features**: AGE_HIGH, CRIM_HIGH categorical variables
- **Distance Metrics**: DIS_SCALED for normalized accessibility

### Model Architecture
- **Deep Neural Network**: Optimized architecture with dropout regularization
- **Batch Normalization**: Improved training stability
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

### Analysis & Visualization
- **Correlation Analysis**: Heatmap of feature relationships
- **Model Comparison**: Side-by-side performance visualization
- **Feature Importance**: Clear ranking of predictive features
- **Stability Testing**: Multi-run validation for reliability

## 📈 Performance Metrics

The improved model demonstrates significant enhancements:

- **Mean Squared Error**: Reduced by approximately 20%
- **R-squared Score**: Notable improvement in explained variance
- **Training Stability**: Consistent results across multiple runs
- **Feature Utilization**: Better leveraging of engineered features

## 🛠️ Dependencies & Requirements

### System Requirements
- **Python**: 3.8+ (tested on 3.8, 3.9, 3.10, 3.11, 3.12)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 100MB free space for models and results

### Core Dependencies
- **TensorFlow**: ≥2.13.0 - Deep learning framework
- **scikit-learn**: ≥1.3.0 - Machine learning utilities  
- **pandas**: ≥2.0.0 - Data manipulation and analysis
- **numpy**: ≥1.24.0 - Numerical computing
- **matplotlib**: ≥3.7.0 - Data visualization
- **seaborn**: ≥0.12.0 - Statistical plotting
- **jupyter**: ≥1.0.0 - Interactive notebook support

### Optional Features
- **Git**: For version control and cloning
- **Make**: For automated build commands (optional)

## 📚 Usage Examples

### Run Full Analysis
```bash
python boston_housing_improved.py
```

### Check Environment
```bash
python check_setup.py
```

### View Results
```bash
# Pretty print results
python -m json.tool results/improvement_results.json

# View summary
cat results/improvement_summary.txt

# Open visualizations (macOS)
open visualizations/*.png
```

### Jupyter Notebook
```bash
jupyter notebook boston_housing_improved.ipynb
```

## 🔍 Key Improvements

1. **Feature Engineering**: 8 carefully crafted features based on domain knowledge
2. **Outlier Handling**: Statistical outlier detection and removal
3. **Model Architecture**: Optimized neural network with regularization
4. **Validation Strategy**: Multi-run analysis for performance stability
5. **Visualization Suite**: Comprehensive plots for model interpretation

## 📊 Expected Output

When you run the project, expect to see:
- Dataset loading and preprocessing logs
- Feature engineering progress
- Model training with validation metrics
- Generated visualizations saved to `visualizations/`
- Model artifacts saved to `models/`
- Detailed results exported to `results/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏆 Acknowledgments

- Boston Housing dataset from the UCI Machine Learning Repository
- TensorFlow and scikit-learn communities
- Inspiration from various machine learning research papers

---

**🚀 Ready to explore advanced machine learning techniques with real-world data!**
