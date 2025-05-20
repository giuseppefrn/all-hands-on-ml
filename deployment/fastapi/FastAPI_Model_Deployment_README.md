
### How to Serve an ML Model Using FastAPI

#### 1. Train and Save Your Model (Example)
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model.pkl")
```

#### 2. Use Provided FastAPI Script
Make sure `model.pkl` is in the same folder as `serve_model_fastapi.py`.

#### 3. Build Docker Image
```bash
docker build -t fastapi-model-server .
```

#### 4. Run the Container
```bash
docker run --rm -p 8000:8000 fastapi-model-server
```

By default, FastAPI will serve at http://0.0.0.0:8000

#### 5. Test with `curl`
```bash
curl curl -X POST http://0.0.0.0:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [5.1, 3.5, 1.4, 0.2]}'
```

Or with multiple samples:
```bash
-d '{"data": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}'
```

#### 6. Swagger UI for Testing
Visit: http://0.0.0.0:8000/docs

FastAPI automatically provides a web interface to test your model.
