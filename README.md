# price-predictions-for-Dallas-and-Gurugram-
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
def generate_dallas_data(n=1200, seed=42):
    np.random.seed(seed)
    neighborhoods = {
        "Uptown":        550_000,
        "Highland Park": 900_000,
        "Plano":         420_000,
        "Frisco":        480_000,
        "Oak Cliff":     280_000,
        "Garland":       250_000,
    }
    records = []
    per_hood = n // len(neighborhoods)
    for hood, base in neighborhoods.items():
        for _ in range(per_hood):
            sqft          = max(600,  np.random.normal(2000, 500))
            bedrooms      = int(np.clip(np.random.normal(3, 1), 1, 7))
            bathrooms     = int(np.clip(np.random.normal(2, 1), 1, 5))
            age           = max(0, int(np.random.exponential(10)))
            school_rating = round(np.clip(np.random.normal(5, 2), 1, 10), 1)
            dist_downtown = round(max(0.5, np.random.normal(12, 6)), 1)
            has_pool      = int(np.random.random() < 0.2)
            garage        = int(np.clip(np.random.poisson(1.5), 0, 4))
            price = (base
                     + sqft          * 150
                     + bedrooms      * 10_000
                     + bathrooms     *  7_000
                     - age           *  2_000
                     + school_rating * 10_000
                     - dist_downtown *  3_000
                     + has_pool      * 30_000
                     + garage        *  6_000
                     + np.random.normal(0, 25_000))
            records.append({
                "neighborhood" : hood,
                "sqft"         : round(sqft),
                "bedrooms"     : bedrooms,
                "bathrooms"    : bathrooms,
                "age_years"    : age,
                "school_rating": school_rating,
                "dist_downtown": dist_downtown,
                "has_pool"     : has_pool,
                "garage_spaces": garage,
                "price_usd"    : max(50_000, round(price, -2)),
            })
    return pd.DataFrame(records)
    def generate_gurgaon_data(n=1200, seed=99):
    np.random.seed(seed)
    sectors = {
        "Golf Course Road" : 14_000,
        "DLF Phase 1-3"    : 12_000,
        "Sohna Road"       :  7_500,
        "New Gurgaon"      :  6_500,
        "Palam Vihar"      :  5_500,
        "Manesar"          :  4_500,
    }
    records = []
    per_sector = n // len(sectors)
    for sector, base_psf in sectors.items():
        for _ in range(per_sector):
            sqft       = max(400, np.random.normal(1300, 350))
            bhk        = int(np.clip(np.random.normal(3, 1), 1, 5))
            floor      = int(np.random.randint(0, 31))
            age        = max(0, int(np.random.exponential(5)))
            amenities  = round(np.clip(np.random.normal(6, 2), 1, 10), 1)
            dist_metro = round(max(0.2, np.random.exponential(2.5)), 1)
            parking    = int(np.clip(np.random.poisson(1), 0, 3))
            furnishing = int(np.random.choice([0, 1, 2], p=[0.35, 0.40, 0.25]))
            psf = max(2000, base_psf
                      + amenities  * 200
                      - dist_metro * 300
                      + floor      *  20
                      - age        * 100
                      + furnishing * 400
                      + np.random.normal(0, base_psf * 0.08))
            price = sqft * psf
            records.append({
                "sector"    : sector,
                "sqft"      : round(sqft),
                "bhk"       : bhk,
                "floor"     : floor,
                "age_years" : age,
                "amenities" : amenities,
                "dist_metro": dist_metro,
                "parking"   : parking,
                "furnishing": furnishing,
                "price_inr" : max(1_500_000, round(price, -3)),
            })
    return pd.DataFrame(records)
    #PREPROCESSING THE RAW DATA
def preprocess(df, target_col, cat_cols):
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on train
    X_test_sc  = scaler.transform(X_test)        # only transform on test

    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc,
            scaler, encoders, feature_names)
NEEDS_SCALE = {"Linear Regression", "Neural Network"}

def get_models():
    return {
        "Linear Regression" : LinearRegression(),

        "Random Forest"     : RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=2, random_state=42, n_jobs=-1),

        "Gradient Boosting" : GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42),

        "Neural Network"    : MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam",
            alpha=0.01, max_iter=1000,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=25, random_state=42),
    }
    def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "R2"  : round(r2_score(y_test, y_pred), 4),
        "MAE" : round(mean_absolute_error(y_test, y_pred), 0),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 0),
        "MAPE": round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2),
    }

def train_and_evaluate(label,
                       X_train, X_test, y_train, y_test,
                       X_train_sc, X_test_sc):
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  {'Model':<22} {'R2':>7}  {'MAE':>14}  {'MAPE':>7}")
    print(f"  {'-'*55}")

    results = {}
    for name, model in get_models().items():
        Xtr = X_train_sc if name in NEEDS_SCALE else X_train
        Xte = X_test_sc  if name in NEEDS_SCALE else X_test
        model.fit(Xtr, y_train)
        m = compute_metrics(model, Xte, y_test)
        results[name] = {**m, "model": model}
        print(f"  {name:<22} {m['R2']:>7.3f}  {m['MAE']:>14,.0f}  {m['MAPE']:>6.1f}%")

    best = max(results, key=lambda k: results[k]['R2'])
    print(f"\n  Best model: {best}  (R2={results[best]['R2']})")
    return results
    def predict_price(model_name, results, scaler, feature_names, values):
    row = pd.DataFrame([dict(zip(feature_names, values))])
    model = results[model_name]["model"]
    if model_name in NEEDS_SCALE:
        row = scaler.transform(row)
    return model.predict(row)[0]

if __name__ == "__main__":

    # ── DALLAS ────────────────────────────────────────────────────────────────
    print("\n>>> Generating Dallas, TX data ...")
    df_d = generate_dallas_data(1200)
    print(df_d.head(3).to_string())
    print(f"\n  Rows: {len(df_d)}  |  "
          f"Price: ${df_d.price_usd.min():,.0f} to ${df_d.price_usd.max():,.0f}  |  "
          f"Mean: ${df_d.price_usd.mean():,.0f}")

    (X_tr_d, X_te_d, y_tr_d, y_te_d,
     X_tr_d_sc, X_te_d_sc,
     scaler_d, enc_d, feat_d) = preprocess(df_d, "price_usd", ["neighborhood"])

    res_d = train_and_evaluate("DALLAS, TX (USD)",
                               X_tr_d, X_te_d, y_tr_d, y_te_d,
                               X_tr_d_sc, X_te_d_sc)
# Predict one sample Dallas house
plano_enc    = enc_d["neighborhood"].transform(["Plano"])[0]
# features: neighborhood, sqft, bedrooms, bathrooms, age_years,
#           school_rating, dist_downtown, has_pool, garage_spaces
sample_dallas = [plano_enc, 2200, 3, 2, 8, 7.0, 10.0, 0, 2]

print("\n  Sample: Plano | 2200sqft | 3bed/2bath | age=8 | "
      "school=7.0 | 10mi downtown | no pool | 2-car garage")
print(f"  {'Model':<22}  Predicted Price")
print(f"  {'-'*42}")
for name in res_d:
    p = predict_price(name, res_d, scaler_d, feat_d, sample_dallas)
    print(f"  {name:<22}  ${p:,.0f}")
    # ── GURGAON ───────────────────────────────────────────────────────────────
print("\n\n>>> Generating Gurgaon, Haryana data ...")
df_g = generate_gurgaon_data(1200)
print(df_g.head(3).to_string())
print(f"\n  Rows: {len(df_g)}  |  "
      f"Price: Rs{df_g.price_inr.min()/1e7:.2f}Cr to Rs{df_g.price_inr.max()/1e7:.2f}Cr  |  "
      f"Mean: Rs{df_g.price_inr.mean()/1e7:.2f}Cr")

(X_tr_g, X_te_g, y_tr_g, y_te_g,
 X_tr_g_sc, X_te_g_sc,
 scaler_g, enc_g, feat_g) = preprocess(df_g, "price_inr", ["sector"])

res_g = train_and_evaluate("GURGAON, HARYANA (INR)",
                           X_tr_g, X_te_g, y_tr_g, y_te_g,
                           X_tr_g_sc, X_te_g_sc)

# Predict one sample Gurgaon flat
sohna_enc     = enc_g["sector"].transform(["Sohna Road"])[0]
# features: sector, sqft, bhk, floor, age_years,
#           amenities, dist_metro, parking, furnishing
sample_gurgaon = [sohna_enc, 1400, 3, 8, 4, 7.0, 1.5, 1, 1]

print("\n  Sample: Sohna Road | 1400sqft | 3BHK | Floor-8 | "
      "age=4 | amenities=7 | metro=1.5km | 1 parking | semi-furnished")
print(f"  {'Model':<22}  Predicted Price")
print(f"  {'-'*48}")
for name in res_g:
    p = predict_price(name, res_g, scaler_g, feat_g, sample_gurgaon)
    print(f"  {name:<22}  Rs{p/1e7:.2f}Cr  (Rs {p:,.0f})")

print("\n\nAll done!")
