# calorie-estimation-analysis

### Software Used
- Python (version 3.9 or higher)

### Python Packages Required

The following packages must be installed:

- pandas  
- matplotlib  
- datasets

### Platform

This project was developed and tested on macOS.  
It should also run on Windows or Linux with the same Python setup.

## Section 2: Map of the Documentation

Project Folder Structure:

```
calorie-estimation-analysis/
│
├── DATA/
│
├── SCRIPTS/
│   ├── ingredient_analysis.py
│   ├── load_data.py
│   └── plot_macros.py
│
├── OUTPUT/
│   ├── ingredient_count_distribution.png
│   ├── top_ingredients.png
│   ├── total_calories_distribution.png
│   ├── total_carb_distribution.png
│   ├── total_fat_distribution.png
│   └── total_protein_distribution.png
│
├── README.md
└── LICENSE
```
## Section 3: Instructions for Reproducing Results

### Step 1: Clone the Repository

```bash
git clone https://github.com/purplemorgy/calorie-estimation-analysis
cd calorie-estimation-analysis
```

---

### Step 2: Create a Virtual Environment

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

---

### Step 3: Install Packages

**Base Packages (Required)**

These packages are required to run data preprocessing and exploratory data analysis:

```bash
pip install pandas matplotlib datasets
```




---

### Step 4: Generate EDA plots


Generate EDA visuals by running

```bash
cd SCRIPTS
python plot_macros.py
python ingredient_analysis.py
```

