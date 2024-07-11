from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Load the dataset from an Excel file
def load_products():
    df = pd.read_excel('C:/Users/aditi/OneDrive/Desktop/myntra/prototype/myntra_clone/myntra_sheet.xlsx')
    products = df.to_dict(orient='records')
    return products

@app.route('/')
def home():
    products = load_products()
    return render_template('index.html', products=products)

if __name__ == '__main__':
    app.run(debug=True)
