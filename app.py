import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import settings
try:
    from config.settings import DEFAULT_SETTINGS
except ImportError:
    # Fallback settings if config not available
    DEFAULT_SETTINGS = {
        "train_test_split": 0.8,
        "model_type": "Linear Regression",
        "prediction_days": 7
    }

# Page configuration
st.set_page_config(
    page_title="📈 Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Currency symbols mapping
CURRENCY_SYMBOLS = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'C$',
    'AUD': 'A$',
    'CHF': 'CHF ',
    'CNY': '¥',
    'HKD': 'HK$',
    'SGD': 'S$',
    'KRW': '₩',
    'INR': '₹',
    'BRL': 'R$',
    'MXN': '$',
    'RUB': '₽',
    'ZAR': 'R',
    'SEK': 'kr',
    'NOK': 'kr',
    'DKK': 'kr',
    'PLN': 'zł',
    'CZK': 'Kč',
    'HUF': 'Ft',
    'ILS': '₪',
    'NZD': 'NZ$',
    'THB': '฿',
    'TRY': '₺',
    'IDR': 'Rp',
    'MYR': 'RM',
    'PHP': '₱',
    'VND': '₫'
}

def get_currency_symbol(currency):
    """Get currency symbol from currency code"""
    return CURRENCY_SYMBOLS.get(currency, currency + ' ')

def format_currency(value, currency):
    """Format value with appropriate currency symbol"""
    symbol = get_currency_symbol(currency)
    
    # Format based on currency (some currencies don't use decimals)
    if currency in ['JPY', 'KRW', 'IDR', 'VND']:
        return f"{symbol}{value:,.0f}"
    else:
        return f"{symbol}{value:,.2f}"

def format_large_number(value, currency=None):
    """Format large numbers with K, M, B suffixes"""
    if value >= 1e12:
        formatted = f"{value/1e12:.1f}T"
    elif value >= 1e9:
        formatted = f"{value/1e9:.1f}B"
    elif value >= 1e6:
        formatted = f"{value/1e6:.1f}M"
    elif value >= 1e3:
        formatted = f"{value/1e3:.1f}K"
    else:
        formatted = f"{value:.0f}"
    
    if currency:
        symbol = get_currency_symbol(currency)
        return f"{symbol}{formatted}"
    return formatted

# Simple yfinance Client
class YFinanceClient:
    """Simplified yfinance client with currency support"""
    
    def __init__(self):
        pass
    
    def get_stock_info(self, symbol):
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get currency, default to USD if not found
            currency = info.get('currency', 'USD')
            
            # Get market cap and format it appropriately
            market_cap = info.get('marketCap')
            if market_cap:
                market_cap_formatted = format_large_number(market_cap, currency)
            else:
                market_cap_formatted = 'N/A'
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': market_cap_formatted,
                'market_cap_raw': market_cap,
                'currency': currency,
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
        except Exception as e:
            st.warning(f"Could not fetch detailed info for {symbol}: {str(e)}")
            return {
                'name': symbol, 
                'sector': 'N/A', 
                'industry': 'N/A', 
                'market_cap': 'N/A',
                'market_cap_raw': None,
                'currency': 'USD',
                'exchange': 'N/A',
                'country': 'N/A'
            }
    
    def get_historical_data(self, symbol, period="2y", interval="1d"):
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return None
            
            # Clean column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Remove unnecessary columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_real_time_data(self, symbol):
        """Get real-time stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data.iloc[-1]  # Last available data point
            return None
        except:
            return None

# Technical Indicators Functions
def calculate_moving_average(data, window):
    """Calculate moving average"""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, lower

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = data.ewm(span=fast).mean()
    exp2 = data.ewm(span=slow).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

# Enhanced prediction function with multiple models including SVM
def train_prediction_model(data, model_type="Linear Regression", train_test_split=0.8, days=7):
    """Train prediction model with multiple algorithm options including SVM"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split as sklearn_split
        import warnings
        warnings.filterwarnings('ignore')
        
        # Prepare enhanced features
        data_copy = data.copy()
        
        # Add technical features
        data_copy['Days'] = np.arange(len(data_copy))
        data_copy['Price_MA_5'] = data_copy['Close'].rolling(window=5).mean()
        data_copy['Price_MA_10'] = data_copy['Close'].rolling(window=10).mean()
        data_copy['Price_Std_5'] = data_copy['Close'].rolling(window=5).std()
        data_copy['Volume_MA_5'] = data_copy['Volume'].rolling(window=5).mean()
        data_copy['Price_Change'] = data_copy['Close'].pct_change()
        data_copy['Volume_Change'] = data_copy['Volume'].pct_change()
        
        # Calculate RSI for additional feature
        delta = data_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # Remove NaN values
        data_copy = data_copy.dropna()
        
        # Prepare feature set based on model type
        if model_type == "Linear Regression":
            # Simple features for linear regression
            feature_columns = ['Days', 'Price_MA_5', 'Volume']
        elif model_type == "SVM RBF":
            # Enhanced features for SVM (works well with normalized features)
            feature_columns = [
                'Days', 'Open', 'High', 'Low', 'Volume',
                'Price_MA_5', 'Price_MA_10', 'Price_Change', 'RSI'
            ]
        else:
            # Enhanced features for tree-based models
            feature_columns = [
                'Days', 'Open', 'High', 'Low', 'Volume',
                'Price_MA_5', 'Price_MA_10', 'Price_Std_5', 
                'Volume_MA_5', 'Price_Change', 'Volume_Change', 'RSI'
            ]
        
        # Use more data for training (minimum 50, maximum 200)
        train_size = min(max(int(len(data_copy) * 0.8), 50), 200)
        train_data = data_copy.tail(train_size).copy()
        
        X = train_data[feature_columns].values
        y = train_data['Close'].values
        
        # Split data for validation
        X_train, X_test, y_train, y_test = sklearn_split(
            X, y, test_size=(1-train_test_split), random_state=42, shuffle=False
        )
        
        # Initialize scaler for SVM (SVM requires feature scaling)
        scaler = None
        if model_type == "SVM RBF":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Initialize model based on selection
        if model_type == "Linear Regression":
            model = LinearRegression()
            model_params = "Default parameters"
            
        elif model_type == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model_params = "Trees: 100, Max Depth: 10"
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model_params = "Trees: 100, Learning Rate: 0.1, Max Depth: 6"
            
        elif model_type == "SVM RBF":
            model = SVR(
                kernel='rbf',
                C=100.0,
                gamma='scale',
                epsilon=0.1
            )
            model_params = "RBF Kernel, C: 100.0, Gamma: scale, Epsilon: 0.1"
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        # Also calculate training metrics for comparison
        y_pred_train = model.predict(X_train_scaled)
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        # Make future predictions
        predictions = []
        
        # For future predictions, we need to simulate the features
        last_close = train_data['Close'].iloc[-1]
        last_day = train_data['Days'].iloc[-1]
        
        for i in range(1, days + 1):
            if model_type == "Linear Regression":
                # Simple feature prediction for linear regression
                future_features = [
                    last_day + i,  # Days
                    last_close,    # Price_MA_5 (approximation)
                    train_data['Volume'].iloc[-1]  # Volume (last known)
                ]
            elif model_type == "SVM RBF":
                # Enhanced features for SVM
                future_features = [
                    last_day + i,  # Days
                    last_close * 0.999,  # Open (slight gap)
                    last_close * 1.005,  # High (small range)
                    last_close * 0.995,  # Low (small range)
                    train_data['Volume'].mean(),  # Volume (average)
                    last_close,  # Price_MA_5
                    last_close,  # Price_MA_10
                    0.001,  # Price_Change (small positive)
                    50.0    # RSI (neutral)
                ]
            else:
                # More complex feature engineering for tree models
                future_features = [
                    last_day + i,  # Days
                    last_close * 0.999,  # Open (slight gap)
                    last_close * 1.005,  # High (small range)
                    last_close * 0.995,  # Low (small range)
                    train_data['Volume'].mean(),  # Volume (average)
                    last_close,  # Price_MA_5
                    last_close,  # Price_MA_10  
                    train_data['Price_Std_5'].iloc[-1],  # Price_Std_5
                    train_data['Volume_MA_5'].iloc[-1],  # Volume_MA_5
                    0.001,  # Price_Change (small positive)
                    0.0,    # Volume_Change (neutral)
                    50.0    # RSI (neutral)
                ]
            
            future_X = np.array([future_features])
            
            # Scale features if using SVM
            if model_type == "SVM RBF" and scaler is not None:
                future_X_scaled = scaler.transform(future_X)
                pred = model.predict(future_X_scaled)[0]
            else:
                pred = model.predict(future_X)[0]
                
            predictions.append(pred)
            
            # Update last_close for next iteration
            last_close = pred
        
        # Create prediction dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions
        })
        pred_df.set_index('Date', inplace=True)
        
        # Combine all metrics and info
        model_info = {
            'model_type': model_type,
            'model_params': model_params,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(feature_columns),
            'feature_names': feature_columns,
            'scaler_used': scaler is not None
        }
        
        combined_metrics = {
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'model_info': model_info
        }
        
        # Store scaler in model info for future use
        model_with_scaler = {
            'model': model,
            'scaler': scaler
        }
        
        return pred_df, combined_metrics, model_with_scaler
        
    except ImportError as e:
        st.error(f"Required library not installed: {str(e)}")
        st.info("Please install: pip install scikit-learn")
        return None, None, None
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'stock_info' not in st.session_state:
        st.session_state.stock_info = None

def sidebar_configuration():
    """Create sidebar configuration"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Stock Selection
    st.sidebar.markdown("### 📊 Stock Selection")
    
    # Popular stocks from different regions/currencies
    popular_stocks = {
        "🇺🇸 Apple Inc.": "AAPL",
        "🇺🇸 Google (Alphabet)": "GOOGL", 
        "🇺🇸 Microsoft": "MSFT",
        "🇺🇸 Tesla": "TSLA",
        "🇺🇸 Amazon": "AMZN",
        "🇺🇸 Meta (Facebook)": "META",
        "🇺🇸 Netflix": "NFLX",
        "🇺🇸 NVIDIA": "NVDA",
        "🇬🇧 ASML Holding": "ASML",
        "🇫🇷 LVMH": "MC.PA",
        "🇩🇪 SAP SE": "SAP",
        "🇯🇵 Toyota Motor": "TM",
        "🇯🇵 Sony Group": "SONY",
        "🇨🇳 Alibaba": "BABA",
        "🇨🇳 Tencent": "0700.HK",
        "🇨🇦 Shopify": "SHOP",
        "🇦🇺 BHP Group": "BHP",
        "🇰🇷 Samsung Electronics": "005930.KS",
        "🇮🇳 Reliance Industries": "RELIANCE.NS",
        "🇮🇩 Bank Central Asia": "BBCA.JK",
        "🇧🇷 Vale S.A.": "VALE",
        "🪙 Bitcoin": "BTC-USD",
        "🪙 Ethereum": "ETH-USD",
        "Custom": "CUSTOM"
    }
    
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        options=list(popular_stocks.keys()),
        help="Choose a popular stock from different regions or select 'Custom'"
    )
    
    if selected_stock == "Custom":
        stock_symbol = st.sidebar.text_input(
            "Enter Stock Symbol", 
            value="",
            help="Enter any valid stock symbol (e.g., AAPL, BBCA.JK, 0700.HK, BTC-USD)"
        ).upper()
    else:
        stock_symbol = popular_stocks[selected_stock]
        st.sidebar.info(f"Selected: {stock_symbol}")
    
    # Data Period Selection
    st.sidebar.markdown("### 📅 Data Period")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "10 Years": "10y",
        "Max": "max"
    }
    
    selected_period = st.sidebar.selectbox(
        "Select Period",
        options=list(period_options.keys()),
        index=3,  # Default to 1 year
        help="Choose historical data period"
    )
    period = period_options[selected_period]
    
    # Data Interval
    interval_options = {
        "Daily": "1d",
        "Weekly": "1wk", 
        "Monthly": "1mo"
    }
    
    selected_interval = st.sidebar.selectbox(
        "Data Interval",
        options=list(interval_options.keys()),
        index=0,  # Default to daily
        help="Choose data frequency"
    )
    interval = interval_options[selected_interval]
    
    # Model Parameters (updated to include SVM)
    st.sidebar.markdown("### 🤖 Model Parameters")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Linear Regression", "Random Forest", "Gradient Boosting", "SVM RBF"],
        index=0 if DEFAULT_SETTINGS.get('model_type') == "Linear Regression" else 0
    )
    
    train_test_split = st.sidebar.slider(
        "Train/Test Split (%)",
        min_value=60,
        max_value=90,
        value=int(DEFAULT_SETTINGS.get('train_test_split', 0.8) * 100)
    ) / 100
    
    # Prediction Parameters
    st.sidebar.markdown("### 🔮 Prediction Settings")
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=DEFAULT_SETTINGS.get('prediction_days', 7)
    )
    
    # Technical Indicators
    st.sidebar.markdown("### 📈 Technical Indicators")
    show_ma = st.sidebar.checkbox("Moving Averages", value=True)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=False)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)
    
    return {
        'stock_symbol': stock_symbol,
        'period': period,
        'interval': interval,
        'model_type': model_type,
        'train_test_split': train_test_split,
        'prediction_days': prediction_days,
        'indicators': {
            'ma': show_ma,
            'rsi': show_rsi,
            'macd': show_macd,
            'bollinger': show_bollinger
        }
    }

def load_stock_data(config):
    """Load stock data from yfinance"""
    if not config['stock_symbol']:
        st.error("⚠️ Please select or enter a stock symbol")
        return None, None
    
    try:
        with st.spinner(f"📥 Loading data for {config['stock_symbol']}..."):
            # Initialize client
            client = YFinanceClient()
            
            # Get stock info
            stock_info = client.get_stock_info(config['stock_symbol'])
            
            # Fetch historical data
            data = client.get_historical_data(
                config['stock_symbol'], 
                period=config['period'],
                interval=config['interval']
            )
            
            if data is None or data.empty:
                return None, None
            
            # Add technical indicators
            if config['indicators']['ma']:
                data['MA_20'] = calculate_moving_average(data['Close'], 20)
                data['MA_50'] = calculate_moving_average(data['Close'], 50)
            
            if config['indicators']['rsi']:
                data['RSI'] = calculate_rsi(data['Close'])
            
            if config['indicators']['bollinger']:
                data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
            
            if config['indicators']['macd']:
                data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
            
            st.success(f"✅ Successfully loaded {len(data)} data points for {config['stock_symbol']}")
            return data, stock_info
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def display_data_overview(data, symbol, stock_info=None):
    """Display data overview and statistics"""
    if data is None:
        return
    
    st.markdown("## 📊 Data Overview")
    
    # Stock Information
    if stock_info:
        currency = stock_info.get('currency', 'USD')
        
        st.markdown(f"### 🏢 **{stock_info['name']}** ({symbol})")
        
        # Basic info in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Exchange:** {stock_info['exchange']}")
        with col2:
            st.info(f"**Country:** {stock_info['country']}")
        with col3:
            st.info(f"**Currency:** {currency}")
        with col4:
            if stock_info['market_cap'] != 'N/A':
                st.info(f"**Market Cap:** {stock_info['market_cap']}")
            else:
                st.info(f"**Market Cap:** N/A")
        
        # Sector and Industry
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Sector:** {stock_info['sector']}")
        with col2:
            st.info(f"**Industry:** {stock_info['industry']}")
    else:
        currency = 'USD'  # Default fallback
    
    # Price metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
    
    with col1:
        st.metric(
            "Current Price",
            format_currency(current_price, currency),
            f"{format_currency(price_change, currency)} ({price_change_pct:.2f}%)"
        )
    
    with col2:
        period_high = data['High'].max()
        st.metric(
            "Period High",
            format_currency(period_high, currency)
        )
    
    with col3:
        period_low = data['Low'].min()
        st.metric(
            "Period Low",
            format_currency(period_low, currency)
        )
    
    with col4:
        avg_volume = data['Volume'].mean()
        st.metric(
            "Avg Volume",
            f"{avg_volume:,.0f}"
        )

def create_price_chart(data, symbol, indicators, currency='USD'):
    """Create interactive price chart with currency support"""
    if data is None:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{symbol} Stock Price', 'Volume', 'Technical Indicators'],
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Moving averages
    if indicators['ma'] and 'MA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MA_20'],
                name='MA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    name='MA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    if indicators['bollinger'] and 'BB_Upper' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                name='Bollinger Bands',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # Technical indicators (RSI or MACD)
    if indicators['rsi'] and 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    elif indicators['macd'] and 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                name='Signal',
                line=dict(color='red')
            ),
            row=3, col=1
        )
    
    # Update layout
    currency_symbol = get_currency_symbol(currency)
    fig.update_layout(
        title=f'{symbol} Stock Analysis Dashboard ({currency})',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text=f"Price ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Indicator", row=3, col=1)
    
    return fig

def create_prediction_chart(historical_data, predictions, symbol, currency='USD'):
    """Create prediction chart with currency support"""
    if historical_data is None or predictions is None:
        return None
    
    fig = go.Figure()
    
    # Historical data (last 30 days)
    recent_data = historical_data.tail(30)
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Predictions
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions['Predicted_Price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        )
    )
    
    currency_symbol = get_currency_symbol(currency)
    fig.update_layout(
        title=f'{symbol} Price Prediction ({currency})',
        xaxis_title='Date',
        yaxis_title=f'Price ({currency_symbol})',
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction App</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🚀 Welcome to Stock Price Prediction App!</strong><br>
        This application supports multiple currencies and exchanges worldwide!
        Analyze stocks from different markets including US, Europe, Asia, and more.
        Data is fetched directly from Yahoo Finance with automatic currency detection.
        <br><br>
        <strong>New:</strong> SVM with RBF kernel support for enhanced prediction accuracy!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Analysis", 
        "🤖 Model Training", 
        "🔮 Price Prediction", 
        "📈 Performance"
    ])
    
    with tab1:
        st.markdown("## 📊 Stock Data Analysis")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📥 Load Data", type="primary"):
                data, info = load_stock_data(config)
                st.session_state.stock_data = data
                st.session_state.stock_info = info
                st.session_state.data_loaded = True if data is not None else False
                st.session_state.model_trained = False  # Reset model state
        
        if st.session_state.data_loaded and st.session_state.stock_data is not None:
            # Get currency for display
            currency = st.session_state.stock_info.get('currency', 'USD') if st.session_state.stock_info else 'USD'
            
            # Data overview
            display_data_overview(
                st.session_state.stock_data, 
                config['stock_symbol'],
                st.session_state.stock_info
            )
            
            # Price chart
            chart = create_price_chart(
                st.session_state.stock_data, 
                config['stock_symbol'], 
                config['indicators'],
                currency
            )
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Data statistics
            with st.expander("📋 Data Statistics"):
                stats_df = st.session_state.stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
                # Format price columns with currency
                for col in ['Open', 'High', 'Low', 'Close']:
                    stats_df[col] = stats_df[col].apply(lambda x: format_currency(x, currency))
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("👆 Click 'Load Data' to fetch stock data from Yahoo Finance (supports multiple currencies)")
    
    with tab2:
        st.markdown("## 🤖 Model Training")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load stock data first in the Data Analysis tab.")
        else:
            # Display model information
            if config['model_type'] == "SVM RBF":
                st.info("""
                **SVM with RBF Kernel** selected:
                - Excellent for non-linear patterns in stock data
                - Automatically scales features for optimal performance
                - Uses Radial Basis Function kernel for complex relationships
                - Good for capturing market volatility patterns
                """)
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        predictions, metrics, model = train_prediction_model(
                            st.session_state.stock_data, 
                            config['model_type'],
                            config['train_test_split'],
                            config['prediction_days']
                        )
                        
                        if predictions is not None and metrics is not None:
                            st.session_state.predictions = predictions
                            st.session_state.model_metrics = metrics
                            st.session_state.trained_model = model
                            st.session_state.model_trained = True
            
            if st.session_state.model_trained and st.session_state.model_metrics:
                # Get currency for display
                currency = st.session_state.stock_info.get('currency', 'USD') if st.session_state.stock_info else 'USD'
                model_info = st.session_state.model_metrics.get('model_info', {})
                
                st.markdown("### 📊 Model Performance Metrics")
                
                # Display model information
                col1, col2 = st.columns(2)
                with col1:
                    scaling_info = "Yes (StandardScaler)" if model_info.get('scaler_used', False) else "No"
                    st.info(f"""
                    **Model Details:**
                    - Type: {model_info.get('model_type', 'Unknown')}
                    - Parameters: {model_info.get('model_params', 'N/A')}
                    - Features Used: {model_info.get('features_used', 'N/A')}
                    - Feature Scaling: {scaling_info}
                    """)
                
                with col2:
                    st.info(f"""
                    **Training Data:**
                    - Training Samples: {model_info.get('training_samples', 'N/A')}
                    - Test Samples: {model_info.get('test_samples', 'N/A')}
                    - Train/Test Split: {int(config['train_test_split']*100)}%/{int((1-config['train_test_split'])*100)}%
                    """)
                
                # Performance metrics comparison
                st.markdown("#### 🎯 Training vs Test Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "MAE (Test)", 
                        format_currency(st.session_state.model_metrics['test_mae'], currency),
                        f"{format_currency(st.session_state.model_metrics['test_mae'] - st.session_state.model_metrics['train_mae'], currency)} vs Train"
                    )
                
                with col2:
                    st.metric(
                        "RMSE (Test)", 
                        format_currency(st.session_state.model_metrics['test_rmse'], currency),
                        f"{format_currency(st.session_state.model_metrics['test_rmse'] - st.session_state.model_metrics['train_rmse'], currency)} vs Train"
                    )
                
                with col3:
                    r2_diff = st.session_state.model_metrics['test_r2'] - st.session_state.model_metrics['train_r2']
                    st.metric(
                        "R² Score (Test)", 
                        f"{st.session_state.model_metrics['test_r2']:.4f}",
                        f"{r2_diff:+.4f} vs Train"
                    )
                
                # Overfitting detection
                train_r2 = st.session_state.model_metrics['train_r2']
                test_r2 = st.session_state.model_metrics['test_r2']
                r2_gap = train_r2 - test_r2
                
                if r2_gap > 0.2:
                    st.warning(f"⚠️ Potential overfitting detected! Training R² ({train_r2:.3f}) significantly higher than Test R² ({test_r2:.3f})")
                elif r2_gap < 0.05:
                    st.success("✅ Good model generalization - similar performance on training and test data")
                else:
                    st.info("ℹ️ Acceptable generalization gap between training and test performance")
                
                # SVM-specific information
                if config['model_type'] == "SVM RBF":
                    st.markdown("#### 🔬 SVM-Specific Information")
                    st.info("""
                    **SVM RBF Model Characteristics:**
                    - Uses Radial Basis Function kernel for non-linear mapping
                    - Features automatically scaled using StandardScaler
                    - Optimal for capturing complex market patterns
                    - C=100.0 (regularization strength), gamma='scale' (kernel coefficient)
                    - Epsilon=0.1 (tolerance for support vector regression)
                    """)
                
                # Feature importance for tree-based models (not applicable for SVM)
                if config['model_type'] in ["Random Forest", "Gradient Boosting"] and st.session_state.trained_model:
                    try:
                        model_obj = st.session_state.trained_model.get('model', st.session_state.trained_model)
                        feature_importance = model_obj.feature_importances_
                        feature_names = model_info.get('feature_names', [])
                        
                        if len(feature_importance) == len(feature_names):
                            st.markdown("#### 🌟 Feature Importance")
                            
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': feature_importance
                            }).sort_values('Importance', ascending=False)
                            
                            # Create horizontal bar chart
                            fig_importance = px.bar(
                                importance_df.head(8), 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title='Top Features Contributing to Predictions'
                            )
                            fig_importance.update_layout(height=400)
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                    except Exception as e:
                        st.info("Feature importance not available for this model")
                
                # Model success message
                model_type = model_info.get('model_type', 'Model')
                st.success(f"✅ {model_type} trained successfully!")
            else:
                st.info("👆 Click 'Train Model' to build prediction model with your selected algorithm")
    
    with tab3:
        st.markdown("## 🔮 Price Prediction")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first in the Model Training tab.")
        else:
            if st.session_state.predictions is not None:
                # Get currency for display
                currency = st.session_state.stock_info.get('currency', 'USD') if st.session_state.stock_info else 'USD'
                
                st.markdown("### 📈 Future Price Predictions")
                
                # Prediction chart
                pred_chart = create_prediction_chart(
                    st.session_state.stock_data,
                    st.session_state.predictions,
                    config['stock_symbol'],
                    currency
                )
                if pred_chart:
                    st.plotly_chart(pred_chart, use_container_width=True)
                
                # Predictions table
                st.markdown("### 📋 Predicted Prices")
                display_df = st.session_state.predictions.copy()
                
                # Format predictions with currency
                display_df['Predicted_Price_Formatted'] = display_df['Predicted_Price'].apply(
                    lambda x: format_currency(x, currency)
                )
                
                # Display formatted table
                display_table = display_df[['Predicted_Price_Formatted']].copy()
                display_table.columns = [f'Predicted Price ({currency})']
                st.dataframe(display_table, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_pred = display_df['Predicted_Price'].mean()
                    st.metric("Average Prediction", format_currency(avg_pred, currency))
                with col2:
                    max_pred = display_df['Predicted_Price'].max()
                    st.metric("Highest Prediction", format_currency(max_pred, currency))
                with col3:
                    min_pred = display_df['Predicted_Price'].min()
                    st.metric("Lowest Prediction", format_currency(min_pred, currency))
                
                # Download predictions
                csv_df = display_df[['Predicted_Price']].copy()
                csv_df['Currency'] = currency
                csv = csv_df.to_csv()
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"{config['stock_symbol']}_predictions_{currency}.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.markdown("## 📈 Performance Analysis")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first to view performance analysis.")
        else:
            if st.session_state.model_metrics:
                # Get currency for display
                currency = st.session_state.stock_info.get('currency', 'USD') if st.session_state.stock_info else 'USD'
                
                st.markdown("### 📊 Model Evaluation")
                
                # Create metrics DataFrame
                metrics_df = pd.DataFrame([{
                    'Metric': 'Mean Absolute Error (Test)',
                    'Value': format_currency(st.session_state.model_metrics['test_mae'], currency),
                    'Description': 'Average absolute difference between actual and predicted prices on test data'
                }, {
                    'Metric': 'Root Mean Square Error (Test)',
                    'Value': format_currency(st.session_state.model_metrics['test_rmse'], currency),
                    'Description': 'Square root of average squared differences on test data'
                }, {
                    'Metric': 'R² Score (Test)',
                    'Value': f"{st.session_state.model_metrics['test_r2']:.4f}",
                    'Description': 'Proportion of variance explained by the model on unseen data (0-1, higher is better)'
                }, {
                    'Metric': 'Mean Absolute Error (Training)',
                    'Value': format_currency(st.session_state.model_metrics['train_mae'], currency),
                    'Description': 'Average absolute difference on training data'
                }, {
                    'Metric': 'Root Mean Square Error (Training)',
                    'Value': format_currency(st.session_state.model_metrics['train_rmse'], currency),
                    'Description': 'Square root of average squared differences on training data'
                }, {
                    'Metric': 'R² Score (Training)',
                    'Value': f"{st.session_state.model_metrics['train_r2']:.4f}",
                    'Description': 'Proportion of variance explained on training data'
                }])
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Model interpretation
                r2_score = st.session_state.model_metrics['test_r2']
                model_type = st.session_state.model_metrics.get('model_info', {}).get('model_type', 'Model')
                
                if r2_score > 0.8:
                    st.success(f"🎯 Excellent {model_type} performance! (R² > 0.8)")
                elif r2_score > 0.6:
                    st.info(f"✅ Good {model_type} performance (R² > 0.6)")
                elif r2_score > 0.4:
                    st.warning(f"⚠️ Moderate {model_type} performance (R² > 0.4)")
                else:
                    st.error(f"❌ Poor {model_type} performance (R² ≤ 0.4) - consider different model or more data")
                
                # Additional insights
                st.markdown("### 💡 Model Insights")
                
                current_price = st.session_state.stock_data['Close'].iloc[-1]
                mae = st.session_state.model_metrics['test_mae']
                mae_percentage = (mae / current_price) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Error Analysis:**
                    - Current Price: {format_currency(current_price, currency)}
                    - Test MAE: {format_currency(mae, currency)}
                    - MAE as % of Price: {mae_percentage:.2f}%
                    - Model Type: {model_type}
                    """)
                
                with col2:
                    # Prediction confidence based on test R²
                    if r2_score > 0.7:
                        confidence = "High"
                        confidence_color = "🟢"
                    elif r2_score > 0.5:
                        confidence = "Medium"
                        confidence_color = "🟡"
                    else:
                        confidence = "Low"
                        confidence_color = "🔴"
                    
                    train_samples = st.session_state.model_metrics.get('model_info', {}).get('training_samples', 'N/A')
                    features_used = st.session_state.model_metrics.get('model_info', {}).get('features_used', 'N/A')
                    
                    st.info(f"""
                    **Prediction Confidence:**
                    - Level: {confidence_color} {confidence}
                    - Test R² Score: {r2_score:.4f}
                    - Training Samples: {train_samples}
                    - Features Used: {features_used}
                    """)
                
                # Trading recommendation (educational only)
                st.markdown("### 📚 Educational Trading Insights")
                st.warning("""
                **Disclaimer:** This is for educational purposes only and should not be considered as financial advice.
                Always consult with a qualified financial advisor before making investment decisions.
                """)
                
                if st.session_state.predictions is not None:
                    last_price = st.session_state.stock_data['Close'].iloc[-1]
                    next_pred = st.session_state.predictions['Predicted_Price'].iloc[0]
                    price_trend = "Upward 📈" if next_pred > last_price else "Downward 📉"
                    trend_percentage = ((next_pred - last_price) / last_price) * 100
                    
                    st.info(f"""
                    **Short-term Trend Analysis:**
                    - Current Price: {format_currency(last_price, currency)}
                    - Next Day Prediction: {format_currency(next_pred, currency)}
                    - Predicted Trend: {price_trend}
                    - Expected Change: {trend_percentage:+.2f}%
                    """)
    
    # Footer with currency support info
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit and yfinance | "
        "Free stock data from Yahoo Finance | "
        "Supports multiple currencies and global markets | "
        "Enhanced with SVM RBF kernel support"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Additional info about supported markets and models
    with st.expander("🌍 Supported Markets, Currencies & Models"):
        st.markdown("""
        **Supported Stock Exchanges & Currencies:**
        
        **🇺🇸 United States (USD)**
        - NASDAQ, NYSE: AAPL, GOOGL, MSFT, TSLA, etc.
        
        **🇪🇺 Europe**
        - 🇬🇧 London (GBP): LLOY.L, BARC.L
        - 🇩🇪 Frankfurt (EUR): SAP.DE, BMW.DE
        - 🇫🇷 Paris (EUR): MC.PA, OR.PA
        - 🇳🇱 Amsterdam (EUR): ASML.AS, RDSA.AS
        
        **🇦🇺 Asia-Pacific**
        - 🇯🇵 Tokyo (JPY): 7203.T (Toyota), 9984.T (SoftBank)
        - 🇭🇰 Hong Kong (HKD): 0700.HK (Tencent), 0941.HK (China Mobile)
        - 🇦🇺 Australia (AUD): BHP.AX, CBA.AX
        - 🇸🇬 Singapore (SGD): D05.SI, O39.SI
        
        **🌏 Emerging Markets**
        - 🇮🇳 India (INR): RELIANCE.NS, TCS.NS
        - 🇰🇷 South Korea (KRW): 005930.KS (Samsung), 035420.KS (NAVER)
        - 🇧🇷 Brazil (BRL): PETR4.SA, VALE3.SA
        - 🇮🇩 Indonesia (IDR): BBCA.JK, TLKM.JK
        
        **🪙 Cryptocurrencies (USD)**
        - BTC-USD, ETH-USD, ADA-USD, etc.
        
        ---
        
        **🤖 Machine Learning Models:**
        
        **Linear Regression**
        - Fast and simple baseline model
        - Works well for linear trends
        - Minimal computational requirements
        
        **Random Forest**
        - Ensemble of decision trees
        - Good for capturing non-linear patterns
        - Provides feature importance rankings
        
        **Gradient Boosting**
        - Sequential ensemble learning
        - Excellent for complex patterns
        - High accuracy potential
        
        **SVM with RBF Kernel** ⭐ *NEW*
        - Support Vector Machine with Radial Basis Function
        - Excellent for non-linear stock price patterns
        - Automatic feature scaling for optimal performance
        - Captures complex market volatility relationships
        - Uses regularization to prevent overfitting
        
        **💡 Tips:**
        - Use correct suffix for each exchange (e.g., .JK for Indonesia, .NS for India)
        - Currency is automatically detected from Yahoo Finance data
        - All price formatting adapts to the stock's native currency
        - SVM RBF often performs well for volatile stocks with complex patterns
        - Try different models to find the best fit for your specific stock
        """)

if __name__ == "__main__":
    main()