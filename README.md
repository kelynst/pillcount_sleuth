# 💊 pillcount_sleuth  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python project to **simulate and analyze medication pill counts** for adherence tracking.  
Useful for learning healthcare data wrangling, patient compliance monitoring, and visualization.  

> ⚠️ Uses synthetic/demo refill data — not real patient records.  

---

## 📌 Project Overview  
- **Goal** → Track pill usage from refill data, estimate adherence, and flag potential under/overuse.  
- **Approach** → Load synthetic refill data, calculate days covered, compare to prescribed schedule, and chart adherence.  
- **Status** → Educational portfolio project — extendable for real-world pharmacy/RCM workflows.  

---

## 📂 Repo Structure  
```
pillcount_sleuth/
│── sleuth.py            # Main script  
│── demo_refills.csv     # Example synthetic dataset  
│── requirements.txt     # pandas, matplotlib  
│── .gitignore  
│── README.md  
│── outputs/             # Charts get saved here  
```

## 📊 Demo Dataset Columns
- **patient_id** → anonymized ID  
- **medication** → drug name  
- **strength_mg** → strength per tablet (mg)  
- **fill_date** → pharmacy fill date (YYYY-MM-DD)  
- **qty_dispensed** → number of tablets dispensed  
- **days_supply** → intended coverage period  
- **prescribed_daily_dose** → tablets/day  
- **count_check_date** → date of pill count check (blank if none)  
- **observed_pills_remaining** → observed pill count on `count_check_date`  

## ✅ Features
- Import synthetic medication refill CSVs  
- Clean blank/duplicate rows  
- Summarize per-patient adherence metrics (e.g., days late, % on-time refills)  
- Flag “risky” patients with poor refill history  
- Export cleaned + flagged CSVs  
- Save simple charts in `outputs/`

## 📦 Requirements
- Python 3.10+
- `pip`
- Packages listed in requirements.txt

Install manually:
```bash
pip install -r requirements.txt
```

## 🚀 Installation

```bash
git clone https://github.com/kelynst/pillcount_sleuth.git
cd pillcount_sleuth
```

Create virtual environment:
```bash
python3 -m venv .venv
```
Activate it:

macOS/Linux
```bash
source .venv/bin/activate 
```
Windows PowerShell
```bash
.venv\Scripts\Activate 
```
Install dependencies:
```bash
pip install -r requirements.txt 
```

## ▶️ Usage
Run the analysis on demo data:
```bash
python sleuth.py demo_refills.csv
```
Output (example):
```plaintext
✅ Pill Count Analysis Complete
• Input file     : demo_refills.csv
• Patients       : 25
• Avg Adherence  : 87.2%
• Non-compliant  : 6 patients (MPR < 80%)
• Charts saved   : outputs/adherence_trends.png
```

## 🔮 Future Improvements
- Handle multiple medications per patient.
- Export reports to Excel or PDF.
- Add pharmacy-level refill summaries
- Explore integration with FHIR MedicationRequest data.

## 🤝 Contributing
Contributions are welcome!
1. Fork the repo
2. Create a branch (git checkout -b feature-xyz)
3. Commit your changes (git commit -m "Add feature xyz")
4. Push to your fork (git push origin feature-xyz)
5. Submit pull request

## ⚠️ Notes
- Dataset is **synthetic/demo** only.
- This tool is for educational/portfolio purposes only. 

## 📜 License
MIT — see LICENSE.
