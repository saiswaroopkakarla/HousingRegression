from utils import load_data, train_models

df = load_data()
results = train_models(df)

for model, metrics in results.items():
    print(f"{model} -> MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.2f}")
