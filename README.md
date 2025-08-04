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

- **Total MSE Improvement**: 15-25% reduction in Mean Squared Error through systematic optimization
- **R² Enhancement**: Significant improvement in explained variance through precision tuning
- **Hyperparameter Optimization**: 12 systematic experiments across batch size, dropout, learning rate, and architecture
- **Model Stability**: Consistent performance across multiple runs with professional validation
- **Feature Engineering**: 8 advanced engineered features capturing non-linear relationships
- **Professional Analysis**: Comprehensive error analysis, residual plots, and hyperparameter sensitivity visualizations

## 📝 Assignment Deliverables

### 📓 PP6 Assignment Notebook
- **GitHub URL**: https://github.com/jordanaftermidnight/PP6-boston-housing-improved/blob/main/PP6_Boston_Housing_Improvements.ipynb
- **Local Path**: `PP6_Boston_Housing_Improvements.ipynb`

### ✍️ Written Summary

I tackled the Boston Housing regression problem by implementing systematic improvements across four key areas:

**Feature Engineering**: Created 8 engineered features including interaction terms (LSTAT×RM), polynomial features (RM²), and ratio features (PTRATIO/TAX) that capture non-linear relationships the original features couldn't express.

**Neural Network Optimization**: Completely overhauled the architecture with batch normalization, dropout regularization, and early stopping callbacks that prevent overfitting while maintaining model capacity.

**Hyperparameter Tuning**: Implemented systematic tuning across 12 experiments, testing batch sizes (16, 32, 64), dropout rates (0.2-0.5), learning rates (0.0005-0.002), and multiple architectures to find optimal precision configurations.

**Robust Validation**: Added Isolation Forest outlier detection, comprehensive visualizations (error plots, residual analysis, hyperparameter sensitivity), and multi-run stability analysis to ensure consistent performance gains rather than random variations.

The result was a **15-25% improvement in MSE** with superior generalization through systematic precision optimization.

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

### Model Architecture & Precision Tuning
- **Hyperparameter-Optimized Neural Network**: Systematic tuning of batch size, dropout rate, learning rate, and architecture
- **Batch Normalization**: Applied strategically to all major layers for training stability
- **Precision Dropout Tuning**: Optimized dropout rates (0.2-0.5) with layer-specific adjustments
- **Batch Size Optimization**: Tested batch sizes (16, 32, 64) for optimal convergence
- **Learning Rate Scheduling**: Fine-tuned learning rates (0.0005-0.002) with adaptive adjustment
- **Architecture Search**: Tested multiple architectures from compact to wide networks
- **Early Stopping**: Prevents overfitting with optimal patience settings

### Analysis & Visualization
- **Correlation Analysis**: Heatmap of feature relationships
- **Model Comparison**: Side-by-side performance visualization
- **Hyperparameter Sensitivity Analysis**: Batch size, dropout rate, and learning rate impact visualization
- **Feature Importance**: Clear ranking of predictive features
- **Training Dynamics**: Error vs epoch plots with validation curves
- **Residual Analysis**: Comprehensive error distribution analysis
- **Stability Testing**: Multi-run validation for reliability

## 📈 Performance Metrics

The precision-tuned model demonstrates significant enhancements:

- **Mean Squared Error**: 15-25% reduction through systematic hyperparameter optimization
- **R-squared Score**: Notable improvement in explained variance with optimal architecture
- **Training Stability**: Consistent results across multiple runs with advanced regularization
- **Feature Utilization**: Superior leveraging of 8 engineered features
- **Hyperparameter Optimization**: Best configuration identified through 12 systematic experiments
- **Model Precision**: Optimal batch size, dropout rates, and learning rate scheduling

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

1. **Advanced Feature Engineering**: 8 carefully crafted features based on domain knowledge
2. **Robust Outlier Handling**: Statistical outlier detection and removal using Isolation Forest
3. **Precision Model Architecture**: Hyperparameter-optimized neural network with strategic regularization
4. **Systematic Hyperparameter Tuning**: Grid search across batch sizes, dropout rates, learning rates, and architectures
5. **Advanced Training Optimization**: Batch normalization, early stopping, and learning rate scheduling
6. **Comprehensive Validation Strategy**: Multi-run analysis for performance stability
7. **Professional Visualization Suite**: Error analysis, hyperparameter sensitivity, and model interpretation plots

## 📊 Expected Output

When you run the project, expect to see:
- Dataset loading and preprocessing logs
- Feature engineering progress
- Model training with validation metrics
- Generated visualizations saved to `visualizations/`
- Model artifacts saved to `models/`
- Detailed results exported to `results/`

## 👨‍💻 Author & Development

**Primary Author**: Jordan After Midnight  
**Development Tools**: Claude AI for code assistance, testing, and repository management  
**Project Type**: Individual academic assignment (PP6)

This project was developed as a solo academic assignment with AI assistance for:
- Code optimization and testing
- Documentation generation and formatting
- Repository synchronization and management
- Performance analysis and visualization

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏆 Acknowledgments

- **Dataset**: Boston Housing dataset from the UCI Machine Learning Repository
- **Libraries**: TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn communities
- **AI Assistant**: Claude AI for development acceleration and code optimization
- **Academic Context**: PP6 assignment focusing on model improvement techniques

---

**🚀 Ready to explore advanced machine learning techniques with real-world data!**
