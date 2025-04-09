import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

model = joblib.load('house_price_model.pkl')


root = tk.Tk()
root.title("House Price Predictor")


house_type_var = tk.StringVar()
sales_type_var = tk.StringVar()
year_built_var = tk.StringVar()
change_offer_purchase_var = tk.StringVar()
rooms_var = tk.StringVar()
m2_var = tk.StringVar()
zip_code_var = tk.StringVar()
nom_interest_rate_var = tk.StringVar()
dk_ann_infl_rate_var = tk.StringVar()
yield_bonds_var = tk.StringVar()

def predict_price():
    try:
        data = {
            'house_type': [house_type_var.get()],
            'sales_type': [sales_type_var.get()],
            'year_build': [int(year_built_var.get())],
            '%_change_between_offer_and_purchase': [float(change_offer_purchase_var.get())], 
            'no_rooms': [int(rooms_var.get())],
            'sqm': [float(m2_var.get())],
            'zip_code': [int(zip_code_var.get())],
            'nom_interest_rate%': [float(nom_interest_rate_var.get())],
            'dk_ann_infl_rate%': [float(dk_ann_infl_rate_var.get())],
            'yield_on_mortgage_credit_bonds%': [float(yield_bonds_var.get())]
        }

        input_df = pd.DataFrame(data)
        prediction = model.predict(input_df)[0]
        result_label.config(text=f"Predicted Price: {int(prediction):,} DKK")
    
    except Exception as e:
        result_label.config(text=f"Error: {e}")


fields = [
    ("House Type", house_type_var, ["Villa", "Summerhouse", "Townhouse","Farm"]),
    ("Sales Type", sales_type_var, ["regular_sale", "family_sale","other_sale","auction"]),
    ("Year Built", year_built_var),
    ("% Change Offer vs Purchase", change_offer_purchase_var),
    ("Number of Rooms", rooms_var),
    ("Square Meters", m2_var),
    ("Zip Code", zip_code_var),
    ("Nominal Interest Rate %", nom_interest_rate_var),
    ("Annual Inflation Rate %", dk_ann_infl_rate_var),
    ("Yield on Mortgage Bonds %", yield_bonds_var)
]

for idx, (label, var, *options) in enumerate(fields):
    tk.Label(root, text=label).grid(row=idx, column=0, sticky="w", padx=5, pady=5)
    if options:
        ttk.Combobox(root, textvariable=var, values=options[0]).grid(row=idx, column=1, padx=5, pady=5)
    else:
        tk.Entry(root, textvariable=var).grid(row=idx, column=1, padx=5, pady=5)


tk.Button(root, text="Predict Price", command=predict_price).grid(row=len(fields), columnspan=2, pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.grid(row=len(fields)+1, columnspan=2)

root.mainloop()
