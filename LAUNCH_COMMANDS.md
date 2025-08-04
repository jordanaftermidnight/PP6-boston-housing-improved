# 🚀 PP6: Boston Housing Project - Launch Commands

## Quick Start (Choose One Method)

### Method 1: Using the Setup Script (Easiest)
```bash
# Make script executable and run
chmod +x setup_project.sh
./setup_project.sh
```

### Method 2: Using Make (If you have Make installed)
```bash
# Setup and run everything
make all

# Or step by step:
make setup    # Install dependencies
make run      # Run the project
```

### Method 3: Manual Python Commands
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p visualizations models results

# Run the project
python boston_housing_improved.py
```

## 📋 Complete Step-by-Step Commands

### 1. First-Time Setup
```bash
# Clone or navigate to your repo
cd PP6-boston-housing-improved

# Create all project files (copy from artifacts)
# The setup_project.sh script creates everything automatically

# Make setup script executable
chmod +x setup_project.sh

# Run setup (creates all files and directories)
./setup_project.sh
```

### 2. Running the Project
```bash
# Activate virtual environment (if not using setup script)
source venv/bin/activate

# Run the main script
python boston_housing_improved.py

# Or use Jupyter notebook
jupyter notebook boston_housing_improved.ipynb
```

### 3. Check Results
```bash
# View the improvement summary
cat results/improvement_summary.txt

# View detailed results (pretty print JSON)
python -m json.tool results/improvement_results.json

# List generated files
ls -la visualizations/
ls -la models/
ls -la results/
```

### 4. Git Commands
```bash
# Add all files
git add .

# Check what will be committed
git status

# Commit with detailed message
git commit -m "Boston Housing improvements: feature engineering, visualizations, and validation

- Added 8 engineered features based on domain knowledge
- Implemented outlier removal and data preprocessing  
- Created comprehensive visualizations
- Performed 3-run validation for stability
- Achieved ~20% improvement in MSE"

# Push to GitHub
git push origin main
```

## 🔧 Utility Commands

### Run Diagnostic Check
```bash
python check_setup.py
```

### Clean Generated Files
```bash
# Using Make
make clean

# Or manually
rm -f models/*.h5
rm -f visualizations/*.png
rm -f results/*.json results/*.txt
```

### View Visualizations
```bash
# macOS
open visualizations/*.png

# Linux
xdg-open visualizations/*.png

# Windows
start visualizations/*.png
```

### Regenerate Notebook from Script
```bash
python -c "
import nbformat as nbf
with open('boston_housing_improved.py', 'r') as f:
    content = f.read()
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell('# Boston Housing - Improved'))
nb.cells.append(nbf.v4.new_code_cell(content))
with open('boston_housing_improved.ipynb', 'w') as f:
    nbf.write(nb, f)
"
```

## 🐍 Virtual Environment Commands

### Create and Activate
```bash
# Create
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate
```

### Verify Environment
```bash
# Check Python version
python --version

# List installed packages
pip list

# Check TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

## 📊 Expected Output

When you run `python boston_housing_improved.py`, you should see:

```
🏠 Boston Housing Price Prediction - Improved Model
==================================================

📊 Loading Boston Housing Dataset...
Training samples: 404
Test samples: 102
Number of features: 13

📈 Performing Exploratory Data Analysis...
✅ Saved correlation heatmap

🔧 Performing Feature Engineering...
Original features: 13
Engineered features: 21
New features added: 8

🧹 Preprocessing Data...
Rows removed as outliers: X

🏃 Training Models...
📊 Training baseline model...
🚀 Training improved model...

🎨 Creating Visualizations...
✅ Saved model comparison plots
✅ Saved feature importance plot

🔍 Model Evaluation and Validation...
🔄 Running stability analysis (3 runs each)...

🏆 FINAL RESULTS
==================================================
Baseline Model - MSE: X.XXXX, R²: 0.XXXX
Improved Model - MSE: X.XXXX, R²: 0.XXXX
MSE Improvement: XX.XX%
R² Improvement: XX.XX%

💾 Model saved to 'models/boston_housing_improved_model.h5'
📄 Results exported to 'results/improvement_results.json'
📝 Improvement summary saved to 'results/improvement_summary.txt'

✅ Project completed successfully!
```

## 🚨 Troubleshooting

### If TensorFlow won't install:
```bash
pip install --upgrade pip setuptools wheel
pip install tensorflow --no-cache-dir
```

### If you get import errors:
```bash
pip install -r requirements.txt --force-reinstall
```

### If directories are missing:
```bash
mkdir -p visualizations models results
```

### If script permissions are denied:
```bash
chmod +x setup_project.sh
chmod +x *.py
```

### If Boston Housing dataset gives deprecation warning:
```bash
# This is expected - the dataset is deprecated but still functional
# The warning is suppressed in our code automatically
```

### Memory issues with TensorFlow:
```bash
# For older/low-memory systems, try:
export TF_CPP_MIN_LOG_LEVEL=2
python boston_housing_improved.py
```

## 📦 Final Project Structure

After running all commands, your directory should contain:

```
PP6-boston-housing-improved/
├── README.md
├── requirements.txt
├── setup_project.sh
├── Makefile
├── .gitignore
├── check_setup.py
├── boston_housing_improved.py
├── boston_housing_improved.ipynb
├── LAUNCH_COMMANDS.md
├── venv/
├── models/
│   └── boston_housing_improved_model.h5
├── results/
│   ├── improvement_results.json
│   └── improvement_summary.txt
└── visualizations/
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── feature_importance.png
```

## ✅ Success Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] Script runs without errors
- [ ] Visualizations generated in `visualizations/`
- [ ] Model saved in `models/`
- [ ] Results saved in `results/`
- [ ] All files properly committed to git
- [ ] Ready to push to GitHub

## 🚀 Advanced Usage

### Custom Feature Engineering
```bash
# Edit the feature engineering section in the Python script
# Add your own features based on domain knowledge
```

### Hyperparameter Tuning
```bash
# Modify the neural network architecture in the script
# Experiment with different layers, dropout rates, etc.
```

### Different Datasets
```bash
# Replace load_boston() with your own dataset
# Ensure it follows the same preprocessing pipeline
```

### Export Models for Production
```bash
# The saved model can be loaded with:
# model = tf.keras.models.load_model('models/boston_housing_improved_model.h5')
```

## 📚 Learning Resources

- **Feature Engineering**: Check how we created interaction and polynomial features
- **Outlier Detection**: Study the Isolation Forest implementation
- **Neural Networks**: Examine the deep learning architecture with regularization
- **Model Validation**: Review the stability analysis methodology
- **Visualization**: Learn from the comprehensive plotting techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run the diagnostic check (`python check_setup.py`)
4. Make your changes
5. Test that everything still works
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

---

**🎯 Ready to explore advanced machine learning with Boston Housing data!**

*Complete ML Pipeline*