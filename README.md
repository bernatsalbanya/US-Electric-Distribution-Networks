# Dataset Preparation Project

## 📌 Overview
This project is designed to **process, clean, and analyze circuit datasets** for different states in the **US Northeast**. It includes scripts for **data cleaning, transformation, and spatial analysis** using Python libraries like **GeoPandas, Pandas, and NumPy**.

## 📂 Project Structure

```
US-Electric-Distribution-Networks/
│── data/               # Store raw (or small sample) datasets
│── notebooks/          # Jupyter notebooks for exploration (if applicable)
│── scripts/            # Python scripts for data processing
│── src/                # Main module(s) for reusable functions
│── tests/              # Unit tests for functions
│── requirements.txt    # List of dependencies
│── README.md           # Project overview and instructions
│── LICENSE             # Open-source license (if applicable)
│── setup.py            # Packaging the project
│── config.yaml         # Configuration file for parameters
│── Makefile            # Commands for easy execution (optional)
```

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone  https://github.com/bernatsalbanya/US-Electric-Distribution-Networks.git
cd US-Electric-Distribution-Networks
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
conda create --name US-Electric-Distribution-Networks python=3.9
conda activate dataset-prep
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

or if using Conda:
```bash
conda install --file requirements.txt
```

## ⚙️ Configuration

Modify `config.yaml` to adjust file paths, parameters, and processing options. Example:

```yaml
data_paths:
  census_tracts:
    maine: 'tl_2022_23_tract.shp'
  circuits:
    "ME_CMP3": "ME_CMP_Hosting_Capacity_Phase3.shp"
    "ME_CMP1": "ME_CMP_Hosting_Capacity_Phase12.shp"
```

## 📌 Usage

### 1️⃣ Run the Main Processing Script
```bash
python main.py
```

### 2️⃣ Run Unit Tests
```bash
pytest tests/
```

### 3️⃣ Reformat Code (Optional)
If using `black` for code formatting:
```bash
black .
```

## 📊 Features

✅ **Data Cleaning:** Standardizes circuit datasets for different states.  
✅ **Geospatial Processing:** Uses GeoPandas to perform spatial joins with Census Tracts.  
✅ **Voltage Estimation:** Computes voltage levels based on circuit rating.  
✅ **Unit Tests:** Ensures code reliability with automated tests.  
✅ **Configurable Parameters:** Modify settings in `config.yaml`.  

## 📜 License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  

You are free to:
- **Share** — Copy and redistribute the material in any medium or format.  
- **Adapt** — Remix, transform, and build upon the material for any purpose, even commercially.  

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

**Full license details:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

![CC BY 4.0 Badge](https://licensebuttons.net/l/by/4.0/88x31.png)

## 👥 Contributors

- **Bernat Salbanyà Rovira** - *Lead Developer*  
- **Jordi Nin Guerrero** - *Contributor*  
- **Ramon Gras Alomà** - *Contributor*  

## 💡 Future Improvements

- Add logging for better debugging  
- Improve efficiency of spatial operations  
- Expand dataset support for additional states  
