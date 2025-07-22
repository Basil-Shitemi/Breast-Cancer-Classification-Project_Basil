import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========== CONFIGURATION ==========
MODEL_PATH = "logreg_model.joblib"
SCALER_PATH = "scaler.joblib"

# ========== DATA LOADING & MODEL TRAINING ==========
def load_and_prepare_data(path):
    data = pd.read_csv(path)
    data = pd.get_dummies(data, drop_first=True)
    if 'diagnosis_M' not in data.columns:
        raise ValueError("Missing 'diagnosis_M' column after encoding.")
    data['diagnosis_M'] = data['diagnosis_M'].astype(int)
    return data

def split_and_scale_data(data, target_col='diagnosis_M', drop_cols=['id'], test_size=0.2):
    X = data.drop(columns=[target_col] + [col for col in drop_cols if col in data.columns])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist(), X, y

def train_and_save_model(data_path):
    data = load_and_prepare_data(data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, X, y = split_and_scale_data(data)
    model = LogisticRegression(C=0.5, penalty='elasticnet', solver='saga', l1_ratio=0.5,
                               max_iter=5000, random_state=0)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    report = classification_report(y_test, model.predict(X_test_scaled))
    return model, scaler, feature_names, acc, report, model.coef_[0], X, y

# ========== INITIALIZATION ==========
data_path = filedialog.askopenfilename(title="Select Breast Cancer CSV", filetypes=[["CSV Files", "*.csv"]])
if not data_path:
    raise SystemExit("No data file selected.")

model, scaler, feature_names, accuracy, report, coef_values, full_X, full_y = train_and_save_model(data_path)

# ========== GUI SETUP ==========
root = tk.Tk()
root.title("Breast Cancer Predictor")
root.geometry("1600x1000")

frame_top = tk.Frame(root)
frame_top.pack(pady=5)

input_vars = {}
rows = [tk.Frame(frame_top) for _ in range(4)]
for row in rows:
    row.pack(fill="x", pady=5)

fields_per_row = (len(feature_names) + 3) // 4

for idx, name in enumerate(feature_names):
    row_idx = idx // fields_per_row
    col_frame = tk.Frame(rows[row_idx])
    col_frame.pack(side="left", padx=3)
    tk.Label(col_frame, text=name, wraplength=150, justify="center").pack()
    var = tk.StringVar(value="0")
    entry = tk.Entry(col_frame, textvariable=var, width=15)
    entry.pack()
    input_vars[name] = var

# ========== Output and Feature Info ==========
frame_center = tk.Frame(root)
frame_center.pack(pady=10, fill="both", expand=True)

output_text = tk.Text(frame_center, height=15, wrap="word")
output_scroll = tk.Scrollbar(frame_center, command=output_text.yview)
output_text.configure(yscrollcommand=output_scroll.set)

output_text.pack(side="left", fill="both", expand=True)
output_scroll.pack(side="right", fill="y")

output_text.insert("1.0", f"Model Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}\n")

# ========== Visualization ==========
def show_eda():
    eda_window = tk.Toplevel(root)
    eda_window.title("Exploratory Data Analysis")
    eda_window.geometry("1200x800")

    canvas_frame = tk.Frame(eda_window)
    canvas_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(canvas_frame)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    tk.Label(scrollable_frame, text="Select Feature for Distribution").pack(pady=5)
    selected_feature = tk.StringVar(value=feature_names[0])
    feature_dropdown = ttk.Combobox(scrollable_frame, textvariable=selected_feature, values=feature_names)
    feature_dropdown.pack(pady=5)

    def plot_selected():
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        sns.histplot(full_X[feature_names][selected_feature.get()][full_y == 1], bins=20, color='red', label='Malignant', ax=axs[0], kde=True)
        sns.histplot(full_X[feature_names][selected_feature.get()][full_y == 0], bins=20, color='green', label='Benign', ax=axs[0], kde=True)
        axs[0].set_title(f"Distribution of {selected_feature.get()}")
        axs[0].legend()

        corr_matrix = full_X.corr()
        sns.heatmap(corr_matrix, ax=axs[1], cmap='coolwarm', cbar=True,
                    xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                    annot=False, fmt=".2f")
        axs[1].set_title("Correlation Heatmap")

        fig.tight_layout()
        canvas_plot = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    tk.Button(scrollable_frame, text="Plot", command=plot_selected).pack(pady=10)

# ========== Functions ==========
def predict():
    try:
        values = []
        for name in feature_names:
            val = float(input_vars[name].get())
            values.append(val)
        scaled_values = scaler.transform([values])
        prob = model.predict_proba(scaled_values)[0][1]
        pred = model.predict(scaled_values)[0]
        label = "Malignant" if pred == 1 else "Benign"
        summary = "\n".join([f"{k}: {v.get()}" for k, v in input_vars.items()])

        top_features = sorted(zip(feature_names, coef_values), key=lambda x: abs(x[1]), reverse=True)[:5]
        impact_text = "\n".join([f"{name}: {weight:.4f}" for name, weight in top_features])

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Input Values:\n{summary}\n\nPrediction: {label} ({prob:.2%} probability)\n\nTop Influential Features:\n{impact_text}\n")
    except ValueError as e:
        messagebox.showerror("Input Error", f"Please enter valid numbers only.\n{e}")
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

def reset():
    for var in input_vars.values():
        var.set("0")
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Model Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}\n")

def close():
    root.destroy()

def mock_data():
    single_obs = [[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
                   1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
                   25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]]
    for i, name in enumerate(feature_names):
        if i < len(single_obs[0]):
            input_vars[name].set(str(single_obs[0][i]))
    predict()

# ========== Bottom Buttons ==========
frame_bottom = tk.Frame(root)
frame_bottom.pack(pady=20)

tk.Button(frame_bottom, text="Predict", command=predict, width=15).grid(row=0, column=0, padx=10)
tk.Button(frame_bottom, text="Reset", command=reset, width=15).grid(row=0, column=1, padx=10)
tk.Button(frame_bottom, text="Close", command=close, width=15).grid(row=0, column=2, padx=10)
tk.Button(frame_bottom, text="Mock Data", command=mock_data, width=15).grid(row=0, column=3, padx=10)
tk.Button(frame_bottom, text="Show EDA", command=show_eda, width=15).grid(row=0, column=4, padx=10)

root.mainloop()
