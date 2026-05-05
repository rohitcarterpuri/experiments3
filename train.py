from preprocess import load_and_preprocess
from model import build_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess("data/housing.csv")

    model = build_model(X_train.shape[1])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, MAE: {mae}")

if __name__ == "__main__":
    main()
