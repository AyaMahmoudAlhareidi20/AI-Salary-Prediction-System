import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd


# -----------------------------
# Load Models
# -----------------------------
reg_model = joblib.load("best_reg_model.pkl")
clf_model = joblib.load("best_clf_model.pkl")


# -----------------------------
# Predict Function
# -----------------------------
def predict():
    try:
        df = pd.DataFrame([{
            "work_year": int(work_year_entry.get()),
            "experience_level": exp_level_var.get(),
            "employment_type": emp_type_var.get(),
            "job_title": job_title_entry.get(),
            "salary": float(salary_entry.get()),
            "salary_currency": currency_entry.get(),
            "employee_residence": residence_entry.get(),
            "remote_ratio": int(remote_entry.get()),
            "company_location": location_entry.get(),
            "company_size": company_size_var.get(),
        }])

        if model_var.get() == "Regression Model":
            pred = reg_model.predict(df)[0]
            result_label.config(text=f"Predicted Salary: ${pred:,.0f}", fg="#00c853")

        else:
            pred = clf_model.predict(df)[0]
            result = "HIGH" if pred == 1 else "LOW"
            color = "#00c853" if pred == 1 else "#d50000"
            result_label.config(text=f"Salary Level: {result}", fg=color)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -----------------------------
# GUI Design
# -----------------------------
root = tk.Tk()
root.title("AI Salary Prediction System")
root.geometry("650x650")
root.configure(bg="#1e1e2e")

title = tk.Label(root, text="AI Salary Prediction", font=("Segoe UI", 22, "bold"), fg="white", bg="#1e1e2e")
title.pack(pady=10)

# Frame
frame = tk.Frame(root, bg="#2a2a3d", padx=20, pady=20)
frame.pack(pady=10)

# Dropdown Model
model_var = tk.StringVar(value="Regression Model")
ttk.Label(frame, text="Select Model:", background="#2a2a3d", foreground="white").grid(row=0, column=0)
ttk.OptionMenu(frame, model_var, "Regression Model", "Regression Model", "Classification Model").grid(row=0, column=1)

# Input Fields
labels = ["Work Year", "Experience Level", "Employment Type", "Job Title", "Salary",
          "Salary Currency", "Employee Residence", "Remote Ratio", "Company Location", "Company Size"]
entries = {}

work_year_entry = tk.Entry(frame); exp_level_var = tk.StringVar()
emp_type_var = tk.StringVar(); job_title_entry = tk.Entry(frame)
salary_entry = tk.Entry(frame); currency_entry = tk.Entry(frame)
residence_entry = tk.Entry(frame); remote_entry = tk.Entry(frame)
location_entry = tk.Entry(frame); company_size_var = tk.StringVar()

fields = [
    ("Work Year", work_year_entry),
    ("Experience Level", exp_level_var, ["EN", "MI", "SE", "EX"]),
    ("Employment Type", emp_type_var, ["FT", "PT", "CT", "FL"]),
    ("Job Title", job_title_entry),
    ("Salary", salary_entry),
    ("Salary Currency", currency_entry),
    ("Employee Residence", residence_entry),
    ("Remote Ratio", remote_entry),
    ("Company Location", location_entry),
    ("Company Size", company_size_var, ["S", "M", "L"]),
]

row = 1
for label, widget, *options in fields:
    tk.Label(frame, text=label + ":", bg="#2a2a3d", fg="white").grid(row=row, column=0, pady=4, sticky="w")
    if options:
        ttk.OptionMenu(frame, widget, options[0][0], *options[0]).grid(row=row, column=1)
    else:
        widget.grid(row=row, column=1)
    row += 1

# Predict Button
predict_btn = tk.Button(root, text="Predict", command=predict,
                        bg="#00bfa5", fg="white", font=("Segoe UI", 14, "bold"),
                        width=20)
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="", font=("Segoe UI", 18, "bold"), bg="#1e1e2e")
result_label.pack(pady=10)

root.mainloop()
