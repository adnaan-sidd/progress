import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_score
import joblib

def calculate_indicators(df):
    df['returns'] = df['Close'].pct_change()
    df['sma20'] = df['Close'].rolling(20).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['upper_bb'] = sma20 + 2*std20
    df['lower_bb'] = sma20 - 2*std20
    
    # Lagged features
    for lag in [1, 2, 3]:
        df[f'returns_lag{lag}'] = df['returns'].shift(lag)
    
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

def feature_engineering(X):
    return X[['sma20', 'rsi', 'macd', 'upper_bb', 'lower_bb',
             'returns_lag1', 'returns_lag2', 'returns_lag3']]

def train_model():
    data = pd.concat([pd.read_csv(f'historical_data/{f}') 
                    for f in os.listdir('historical_data') if f.endswith('.csv')])
    
    data = calculate_indicators(data)
    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    
    # Corrected ColumnTransformer
    preprocessor = ColumnTransformer([
        ('imputer', SimpleImputer(strategy='median'), ['sma20', 'rsi', 'macd', 'upper_bb', 'lower_bb']),
        ('scaler', StandardScaler(), ['sma20', 'rsi', 'macd', 'upper_bb', 'lower_bb'])
    ])
    
    pipeline = Pipeline([
        ('feature_selector', FunctionTransformer(feature_engineering)),
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'classifier__max_depth': [3, 4],
        'classifier__learning_rate': [0.05, 0.1]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='precision',
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting model training...")
    grid_search.fit(train, train['target'])
    
    best_model = grid_search.best_estimator_
    test_preds = best_model.predict(test)
    print("\nBest Model Evaluation:")
    print(classification_report(test['target'], test_preds))
    print(f"Best parameters: {grid_search.best_params_}")
    
    joblib.dump(best_model, 'eurusd_model.pkl')
    print("Model saved successfully!")

if __name__ == '__main__':
    train_model()
