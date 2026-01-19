import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path, sheet):
    """Load excel sheet."""
    return pd.read_excel(path, sheet_name=sheet)

def show_info(table):
    print("\nColumns:", list(table.columns))
    print("Missing values:\n", table.isna().sum())
    print("Stats:\n", table.describe())

# A1)
def split_features_target(table, feature_cols, target_col):
    features = table[feature_cols].to_numpy()
    target = table[target_col].to_numpy()
    return features, target

def get_rank(features):
    return np.linalg.matrix_rank(features)

def get_cost(features, target):
    return np.linalg.pinv(features) @ target

# A2)
def mark_rich_or_poor(table, target_col, limit=200):
    table["Class"] = ["RICH" if p > limit else "POOR" for p in table[target_col]]
    return table

# A3)
def mean(arr):
    return sum(arr) / len(arr)

def var(arr):
    m = mean(arr)
    return sum((x-m)**2 for x in arr) / len(arr)

def pick_wednesday(table):
    table["weekday"] = pd.to_datetime(table["Date"]).dt.day_name()
    return table[table["weekday"] == "Wednesday"]

def pick_month(table, month=4):
    table["month"] = pd.to_datetime(table["Date"]).dt.month
    return table[table["month"] == month]

def prob_negative(arr):
    return np.sum(arr < 0) / len(arr)

def add_weekday(t):
    t["weekday"] = pd.to_datetime(t["Date"]).dt.day_name()
    return t

def prob_wed_profit(t):
    wed = t[t["weekday"] == "Wednesday"].copy()
    if len(wed) == 0:
        return 0
    return sum(wed["Chg%"] > 0) / len(wed)

def cond_wed_profit(t):
    return prob_wed_profit(t)

def plot_wday_vs_chg(t):
    plt.scatter(t["weekday"], t["Chg%"])
    plt.title("Chg% vs Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Chg%")
    plt.show()

# A5), A6), A7)
def jaccard(a, b):
    f11 = np.sum((a==1)&(b==1))
    f10 = np.sum((a==1)&(b==0))
    f01 = np.sum((a==0)&(b==1))
    denom = f11 + f10 + f01
    return f11/denom if denom != 0 else 0

def smc(a, b):
    total = len(a)
    matches = np.sum(a==b)
    return matches/total if total != 0 else 0

def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0   
    return np.dot(a, b) / (na * nb)

def compare_similarity(data):
    """Use first 2 rows."""
    a = data[0]
    b = data[1]
    print("Jaccard:", jaccard(a,b))
    print("SMC:", smc(a,b))
    print("Cosine:", cosine(a,b))

def make_heatmap(data, title):
    sns.heatmap(data, annot=False)
    plt.title(title)
    plt.show()

def make_pairwise(data, measure="cosine"):
    data = data[:20]   
    n = len(data)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if measure=="cosine":
                mat[i,j] = cosine(data[i], data[j])
            elif measure=="jaccard":
                mat[i,j] = jaccard(data[i], data[j])
            else:
                mat[i,j] = smc(data[i], data[j])
    return mat

# A8)
def fill_missing(table):
    for col in table.columns:
        if table[col].dtype == 'object':
            table[col] = table[col].fillna(table[col].mode()[0])
        else:
            table[col] = table[col].fillna(table[col].median())
    return table

# A9)
def normalize(table, cols):
    for col in cols:
        table[col]=(table[col]-table[col].min())/(table[col].max()-table[col].min())
    return table

# O1–O3 (optional)
def make_square_sets(table, size=5):
    a = table.iloc[:size,:size].to_numpy()
    b = table.iloc[-size:,-size:].to_numpy()
    print("O1 squares made.")
    return a,b

def random_similarity(table, size=20):
    numeric = table.select_dtypes(include=[np.number])
    binary = (numeric > numeric.median()).astype(int)
    binary = binary.loc[:, binary.nunique() > 1]
    sample = binary.sample(size)
    make_heatmap(make_pairwise(sample.values,"cosine"), "Random Cosine (O2)")
    make_heatmap(make_pairwise(sample.values,"jaccard"), "Random Jaccard (O2)")

def marketing_similarity(table):
    numeric = table.select_dtypes(include=[np.number])
    binary = (numeric > numeric.median()).astype(int)
    compare_similarity(binary.values)
    make_heatmap(make_pairwise(binary.values,"cosine"), "Marketing Cosine (O3)")

def main():
    file_path = "Lab Session Data.xlsx"
    purchase = load_data(file_path,"Purchase data")
    feats = ["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]
    target = "Payment (Rs)"
    fmat, payments = split_features_target(purchase, feats, target)
    print("Matrix Rank:", get_rank(fmat))
    print("Product Cost:", get_cost(fmat, payments))
    
    purchase = mark_rich_or_poor(purchase, target)
    print(purchase[["Payment (Rs)","Class"]])

    irctc = load_data(file_path, "IRCTC Stock Price")
    irctc = add_weekday(irctc)

    prices = irctc["Price"].values
    print("Mean:", mean(prices), "Var:", var(prices))
    print("Prob loss overall:", prob_negative(irctc["Chg%"].values))

    print("P(profit on Wed):", prob_wed_profit(irctc))
    print("P(profit | Wed):", cond_wed_profit(irctc))

    plot_wday_vs_chg(irctc)
    
    thyroid = load_data(file_path,"thyroid0387_UCI")
    show_info(thyroid)
    
    nums = thyroid.select_dtypes(include=[np.number])
    binary = (nums > nums.median()).astype(int)
    compare_similarity(binary.values)
    make_heatmap(make_pairwise(binary.values,"cosine"),"Thyroid Cosine")

    thyroid = fill_missing(thyroid)
    
    numeric_cols = thyroid.select_dtypes(include=[np.number]).columns
    thyroid = normalize(thyroid, numeric_cols)

    make_square_sets(purchase)

    random_similarity(thyroid)

    marketing = load_data(file_path,"marketing_campaign")
    marketing_similarity(marketing)

if __name__=="__main__":
    main()

