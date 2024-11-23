from fabric_predictor import FabricPredictor

if __name__ == "__main__":
    predictor = FabricPredictor()
    predictor.train_model()
    print("Model training completed!") 