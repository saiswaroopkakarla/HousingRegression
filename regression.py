from utils import load_data, train_models, tune_and_train_models

df = load_data()
results = train_models(df)

for model, metrics in results.items():
    print(f"{model} -> MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.2f}")


df = load_data()
results = tune_and_train_models(df)

for model, metrics in results.items():
    print(f"{model} ->")
    print(f"  Best Params: {metrics['Best Params']}")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  R²: {metrics['R2']:.2f}")

