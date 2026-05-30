from flask import Flask, request, render_template, jsonify
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)

# Stock symbols
STOCKS = {
    "amazon": "AMZN", "apple": "AAPL", "google": "GOOGL", "meta": "META",
    "microsoft": "MSFT", "netflix": "NFLX", "nvidia": "NVDA"
}

def fetch_live_stock_data(symbol):
    """Fetch live stock data from Yahoo Finance for 1 year"""
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y", interval="1d")  
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  
    return df

def get_live_price(symbol):
    """Fetch the latest stock price"""
    stock = yf.Ticker(symbol)
    live_price = stock.history(period="1d")['Close'][0]  
    return round(live_price, 2)

def get_prediction_models(df):
    """Train prediction models using historical stock data"""
    df = df.copy()
    df['Day'] = np.arange(len(df))
    X = df[['Day']].values
    Y = df['Close'].values
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    for model in models.values():
        model.fit(X, Y)
    return models, X, Y

def generate_stock_plots(df, stock_name):
    """Generate interactive stock analysis charts"""
    df = df.copy()
    df['Daily Return'] = df['Close'].pct_change()
    df['Moving Average'] = df['Close'].rolling(window=10).mean()
    models, X_train, Y_train = get_prediction_models(df)

    df['Date'] = pd.to_datetime(df['Date'])
    tick_step = max(len(df) // 10, 1)  
    tickvals = df['Date'][::tick_step]
    ticktext = df['Date'][::tick_step].dt.strftime('%Y-%m-%d')

    plot_layout = go.Layout(
        title=f'{stock_name} Stock Price Analysis',
        width=1200,
        height=600,
        xaxis=dict(
            title='Date',
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
        ),
        yaxis=dict(
            title='Price (USD)',
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=True,
            showline=True,
            linecolor='black',
            ticks='outside',
        ),
        plot_bgcolor='rgba(0, 0, 0, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0.1)',
        font=dict(color='black'),
        hovermode='closest',
        showlegend=True,
    )

    plots = {
        'price': go.Figure(
            data=[go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=f'{stock_name} Close Price', line=dict(color='black', width=2))],
            layout=plot_layout
        ).to_html(full_html=False),
        'volume': go.Figure(
            data=[go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker=dict(color='orange'))],
            layout=plot_layout
        ).to_html(full_html=False),
        'candlestick': go.Figure(
            data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')],
            layout=plot_layout
        ).to_html(full_html=False),
        'daily_return': go.Figure(
            data=[go.Scatter(x=df['Date'], y=df['Daily Return'], mode='lines', name='Daily Returns', line=dict(color='magenta', width=2))],
            layout=plot_layout
        ).to_html(full_html=False),
        'moving_average': go.Figure(
            data=[go.Scatter(x=df['Date'], y=df['Moving Average'], mode='lines', name='Moving Average', line=dict(color='black', dash='dot', width=2))],
            layout=plot_layout
        ).to_html(full_html=False),
        'prediction': go.Figure(
            data=[go.Scatter(x=X_train.T[0], y=Y_train, mode='markers', name='Actual', marker=dict(color='green', size=8))] +
                [go.Scatter(x=X_train.T[0], y=model.predict(X_train).T, mode='lines', name=f'{name} Prediction', line=dict(width=2)) for name, model in models.items()],
            layout=plot_layout
        ).to_html(full_html=False)
    }
    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_stock', methods=['POST'])
def submit_stock():
    stock_name = request.form['stockName'].lower()
    if stock_name in STOCKS:
        stock_symbol = STOCKS[stock_name]
        stock_data = fetch_live_stock_data(stock_symbol)
        live_price = get_live_price(stock_symbol)
        plots = generate_stock_plots(stock_data, stock_name.capitalize())
        return render_template('stock_analysis.html', stock_name=stock_name.capitalize(), plots=plots, live_price=live_price)
    return render_template('error.html', message="Stock not found. Please enter a valid stock name.")

@app.route('/get_live_price/<stock_name>')
def get_live_price_api(stock_name):
    stock_symbol = STOCKS.get(stock_name.lower())
    if stock_symbol:
        try:
            live_price = get_live_price(stock_symbol)
            return jsonify({'live_price': live_price})
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Stock not found'})

if __name__ == '__main__':
    app.run(debug=True)
