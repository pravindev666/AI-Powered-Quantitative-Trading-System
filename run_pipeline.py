import argparse
from app import EnhancedNiftyPredictionSystem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pipeline non-interactively')
    parser.add_argument('--days', type=int, default=200)
    parser.add_argument('--no-lstm', action='store_true')
    parser.add_argument('--no-shap', action='store_true')
    parser.add_argument('--force-refresh', action='store_true')
    args = parser.parse_args()

    system = EnhancedNiftyPredictionSystem()
    system.enable_lstm = not args.no_lstm
    system.enable_shap = not args.no_shap

    print('Loading data...')
    system.load_data(days=args.days, force_refresh=args.force_refresh)
    print('Calculating indicators...')
    system.calculate_indicators()
    print('Generating signals...')
    system.generate_signals()
    print('Training ML models (this may take several minutes)...')
    system.train_ml_models()
    print('Predicting next day...')
    rec = system.predict_next_day()
    print('Done. Summary:')
    print(rec)
