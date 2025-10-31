import pandas as pd
import numpy as np
from app import MLEnsemble

# Small deterministic toy dataset
np.random.seed(42)
N = 120
features = [f"f{i}" for i in range(10)]
X = pd.DataFrame(np.random.randn(N, len(features)), columns=features)
# create a weak signal: sum of first two features positive -> class 1
y = ((X['f0'] + X['f1']) > 0).astype(int)

print("Toy dataset prepared:", X.shape)

ml = MLEnsemble()
res = ml.train(X, y, test_size=0.2)
if res is None:
    print("Training did not run (insufficient data or other issue)")
else:
    latest = X.iloc[[-1]]
    pred = ml.predict(latest)
    print("Prediction:", pred)
    ml._print_summary()
