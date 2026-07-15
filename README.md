# Lending-Club-loan-project

## 🛠️ Data Processing & Machine Learning Pipeline

This pipeline extracts, cleans, and optimizes a large-scale loan dataset, culminating in the training of a LightGBM model to evaluate key risk drivers via feature importance.

---

### 📋 Pipeline Workflow

#### 1. Targeted Data Loading & Memory Management
* **Selective Loading**: Loads only the target variable (`loan_status`) and $23$ predictive features from Google Drive, preventing system RAM exhaustion.
* **Downsampling**: Limits the dataset size to a representative sample of $500,000$ rows (downsampled from 1.3M+ rows) to guarantee fast iteration and memory-safe execution.
* **Garbage Collection**: Explicitly frees memory by deleting intermediary dataframes and triggering Python's garbage collector (`gc.collect()`).

#### 2. Target Binary Mapping & Cleaning
* Filters out ongoing or active loans (e.g., *Current*, *In Grace Period*) to focus strictly on historical closed outcomes.
* Maps categorical loan statuses into a clean binary classification scheme:
  * **`0` (Good / Paid)**: Fully Paid, Credit Policy Fully Paid.
  * **`1` (Bad / Defaulted)**: Charged Off, Default, Credit Policy Charged Off.
* Casts the resulting target column to a memory-efficient `int8` data type.

#### 3. Data Type Downcasting & Optimization
To optimize training speed and minimize the model's memory footprint, the features are aggressively downcasted:
* **Floats**: `float64` ──> `float32`
* **Integers**: `int64` ──> `int32`
* **Categoricals**: Identifies and casts text-based object columns explicitly to the pandas `category` data type, preparing them for native categorical handling by LightGBM.

#### 4. LightGBM Model Training
* Constructs native LightGBM `Dataset` objects, feeding the downcasted categorical columns directly into the algorithm.
* Trains a baseline Gradient Boosting Decision Tree (`gbdt`) classifier with a binary classification objective, tracking performance using the Area Under the ROC Curve (`auc`) metric over $100$ boosting rounds.

#### 5. Feature Importance Extraction
* Extracts the cumulative **Information Gain** contributed by each of the $23$ features during the tree-building process.
* Generates and saves a clean horizontal bar chart using `seaborn` and `matplotlib` to rank the variables by predictive power.
