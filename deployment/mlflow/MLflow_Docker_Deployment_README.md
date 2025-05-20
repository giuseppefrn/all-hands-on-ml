
### How to Serve a Model with MLflow Using Docker

#### 1. Export a Model (Example)
Train a model and save it with MLflow:
```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

mlflow.sklearn.save_model(model, path="model")
```

#### 2. Create Dockerfile
Use the provided Dockerfile to containerize the model server.

#### 3. Build Docker Image
```bash
docker build -t mlflow-model-server .
```

#### 4. Run the Container
```bash
docker run -p 1234:1234 mlflow-model-server
```

#### 5. Send a Prediction Request
Prepare JSON input with the same structure as your model expects:
```bash
curl -X POST http://localhost:1234/invocations \
     -H "Content-Type: application/json" \
     -d '{
           "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
           "data": [[5.1, 3.5, 1.4, 0.2]]
         }'
```

#### Output
The API will return prediction results in JSON format.

You can copy the Dockerfile and instructions together with the trained model into the Docker context.
