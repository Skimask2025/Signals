import ccxt
import pandas as pd
import time
import requests
import datetime
import threading
import numpy as np
from typing import Dict, List, Optional
import logging
import json
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging with Windows-compatible format (no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_generator.log'),
        logging.StreamHandler()
    ]
)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
EXCHANGE_NAME = 'kucoin'
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
DATA_FILE = 'signals_history.json'

# Strategy Parameters
FAST_MA_PERIOD = 50
SLOW_MA_PERIOD = 200
ATR_PERIOD = 14
TRAILING_STOP_PERCENT = 0.10
TAKE_PROFIT_MULTIPLIER = 2.0

# Global trade history
trade_history = {
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'active_trade': None,
    'message_ids': {},
    'all_signals': []
}

def load_signals_history():
    """Load signals history from JSON file"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                # Convert string dates back to datetime objects
                for signal in data.get('all_signals', []):
                    if 'timestamp' in signal and isinstance(signal['timestamp'], str):
                        signal['timestamp'] = datetime.datetime.fromisoformat(signal['timestamp'])
                    if 'entry_time' in signal and isinstance(signal['entry_time'], str):
                        signal['entry_time'] = datetime.datetime.fromisoformat(signal['entry_time'])
                    if 'exit_time' in signal and isinstance(signal['exit_time'], str):
                        signal['exit_time'] = datetime.datetime.fromisoformat(signal['exit_time'])
                return data
        except Exception as e:
            logging.error(f"Error loading signals history: {e}")
            logging.error("Creating new signals history file")
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'active_trade': None,
        'message_ids': {},
        'all_signals': []
    }

def save_signals_history():
    """Save signals history to JSON file"""
    try:
        # Convert datetime objects to strings for JSON serialization
        data_to_save = trade_history.copy()
        data_to_save['all_signals'] = []
        
        for signal in trade_history['all_signals']:
            signal_copy = signal.copy()
            if 'timestamp' in signal_copy and isinstance(signal_copy['timestamp'], datetime.datetime):
                signal_copy['timestamp'] = signal_copy['timestamp'].isoformat()
            if 'entry_time' in signal_copy and isinstance(signal_copy['entry_time'], datetime.datetime):
                signal_copy['entry_time'] = signal_copy['entry_time'].isoformat()
            if 'exit_time' in signal_copy and isinstance(signal_copy['exit_time'], datetime.datetime):
                signal_copy['exit_time'] = signal_copy['exit_time'].isoformat()
            data_to_save['all_signals'].append(signal_copy)
        
        with open(DATA_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        logging.info("Signals history saved successfully")
    except Exception as e:
        logging.error(f"Error saving signals history: {e}")

def send_telegram_message(message: str, reply_to_message_id: Optional[int] = None) -> Optional[int]:
    """Send message to Telegram channel"""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN':
        logging.warning("Telegram bot token not configured")
        return None
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    if reply_to_message_id:
        payload['reply_to_message_id'] = reply_to_message_id
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            message_data = response.json()
            return message_data['result']['message_id']
        else:
            logging.error(f"Telegram API error: {response.status_code}")
        return None
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return None

def calculate_accuracy() -> float:
    """Calculate trading accuracy"""
    if trade_history['total_trades'] == 0:
        return 0
    return (trade_history['winning_trades'] / trade_history['total_trades']) * 100

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def fetch_ohlcv_data(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data from exchange"""
    try:
        logging.info(f"Fetching {limit} {timeframe} candles for {symbol}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            logging.warning("No data returned from exchange")
            return pd.DataFrame()
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logging.info(f"Successfully fetched {len(df)} candles")
        logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        logging.info(f"Latest price: ${df['close'].iloc[-1]:.2f}")
        
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def check_trading_signals(df: pd.DataFrame) -> Dict:
    """Check for trading signals based on strategy"""
    signals = {
        'buy_signal': False,
        'sell_signal': False,
        'stop_loss_hit': False,
        'take_profit_hit': False,
        'entry_price': 0,
        'stop_loss': 0,
        'take_profit': 0,
        'current_price': df['close'].iloc[-1] if not df.empty else 0,
        'fast_ma': 0,
        'slow_ma': 0,
        'atr': 0
    }
    
    if len(df) < SLOW_MA_PERIOD:
        logging.warning(f"Not enough data: {len(df)} < {SLOW_MA_PERIOD} (required)")
        return signals
    
    # Calculate indicators
    df['fast_ma'] = calculate_sma(df['close'], FAST_MA_PERIOD)
    df['slow_ma'] = calculate_sma(df['close'], SLOW_MA_PERIOD)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
    
    # Get the last valid values (skip NaN)
    valid_data = df.dropna()
    if len(valid_data) < 2:
        logging.warning("Not enough valid data after dropping NaN")
        return signals
    
    current_fast_ma = valid_data['fast_ma'].iloc[-1]
    current_slow_ma = valid_data['slow_ma'].iloc[-1]
    previous_fast_ma = valid_data['fast_ma'].iloc[-2]
    previous_slow_ma = valid_data['slow_ma'].iloc[-2]
    current_atr = valid_data['atr'].iloc[-1]
    current_price = valid_data['close'].iloc[-1]
    
    signals.update({
        'fast_ma': current_fast_ma,
        'slow_ma': current_slow_ma,
        'atr': current_atr,
        'current_price': current_price
    })
    
    # Log current market conditions
    logging.info("MARKET ANALYSIS:")
    logging.info(f"   Current Price: ${current_price:.2f}")
    logging.info(f"   Fast MA ({FAST_MA_PERIOD}): {current_fast_ma:.2f}")
    logging.info(f"   Slow MA ({SLOW_MA_PERIOD}): {current_slow_ma:.2f}")
    logging.info(f"   ATR ({ATR_PERIOD}): {current_atr:.2f}")
    
    # Check for crossovers
    crossover_up = previous_fast_ma <= previous_slow_ma and current_fast_ma > current_slow_ma
    crossover_down = previous_fast_ma >= previous_slow_ma and current_fast_ma < current_slow_ma
    
    logging.info("   Crossover Analysis:")
    logging.info(f"   - Fast MA > Slow MA: {current_fast_ma > current_slow_ma}")
    logging.info(f"   - Previous Fast MA > Previous Slow MA: {previous_fast_ma > previous_slow_ma}")
    logging.info(f"   - BUY Signal (crossover up): {crossover_up}")
    logging.info(f"   - SELL Signal (crossover down): {crossover_down}")
    
    if crossover_up and not trade_history['active_trade']:
        signals['buy_signal'] = True
        signals['entry_price'] = current_price
        signals['stop_loss'] = current_price * (1 - TRAILING_STOP_PERCENT)
        signals['take_profit'] = current_price + (current_atr * TAKE_PROFIT_MULTIPLIER)
        logging.info("BUY SIGNAL DETECTED!")
    
    # Check for active trade conditions
    if trade_history['active_trade']:
        active_trade = trade_history['active_trade']
        
        logging.info("ACTIVE TRADE:")
        logging.info(f"   Entry Price: ${active_trade['entry_price']:.2f}")
        logging.info(f"   Stop Loss: ${active_trade['stop_loss']:.2f}")
        logging.info(f"   Take Profit: ${active_trade['take_profit']:.2f}")
        logging.info(f"   Current P/L: {((current_price / active_trade['entry_price']) - 1) * 100:.2f}%")
        
        # Check for stop loss hit
        if current_price <= active_trade['stop_loss']:
            signals['stop_loss_hit'] = True
            signals['sell_signal'] = True
            logging.info("STOP LOSS HIT!")
        
        # Check for take profit hit
        elif current_price >= active_trade['take_profit']:
            signals['take_profit_hit'] = True
            signals['sell_signal'] = True
            logging.info("TAKE PROFIT HIT!")
        
        # Check for trend reversal sell
        elif crossover_down:
            signals['sell_signal'] = True
            logging.info("TREND REVERSAL SELL SIGNAL!")
    
    return signals

def process_signal(signals: Dict, df: pd.DataFrame):
    """Process trading signals and send Telegram messages"""
    current_time = df.index[-1]
    current_price = signals['current_price']
    
    if signals['buy_signal'] and not trade_history['active_trade']:
        trade_id = f"{current_time}_{signals['entry_price']}"
        trade_history['active_trade'] = {
            'entry_price': signals['entry_price'],
            'entry_time': current_time,
            'direction': 'LONG',
            'stop_loss': signals['stop_loss'],
            'take_profit': signals['take_profit'],
            'trade_id': trade_id
        }
        trade_history['total_trades'] += 1
        
        # Save signal to history
        signal_record = {
            'type': 'BUY',
            'timestamp': current_time,
            'price': signals['entry_price'],
            'stop_loss': signals['stop_loss'],
            'take_profit': signals['take_profit'],
            'trade_id': trade_id,
            'status': 'OPEN'
        }
        trade_history['all_signals'].append(signal_record)
        
        message = f"🚀 <b>BUY SIGNAL</b> 🚀\n\n"
        message += f"<b>Symbol:</b> {SYMBOL}\n"
        message += f"<b>Entry Price:</b> ${signals['entry_price']:.2f}\n"
        message += f"<b>Stop Loss:</b> ${signals['stop_loss']:.2f}\n"
        message += f"<b>Take Profit:</b> ${signals['take_profit']:.2f}\n"
        message += f"<b>Time:</b> {current_time}\n"
        message += f"<b>Strategy:</b> Trend Following (MA Crossover)\n"
        message += f"<b>Historical Accuracy:</b> {calculate_accuracy():.2f}%"
        message += f"\n<b>Total Signals:</b> {trade_history['total_trades']}"
        
        message_id = send_telegram_message(message)
        if message_id:
            trade_history['message_ids'][trade_id] = message_id
        
        save_signals_history()
    
    elif signals['sell_signal'] and trade_history['active_trade']:
        active_trade = trade_history['active_trade']
        trade_id = active_trade['trade_id']
        
        exit_type = "TREND REVERSAL"
        if signals['stop_loss_hit']:
            exit_type = "STOP LOSS"
        elif signals['take_profit_hit']:
            exit_type = "TAKE PROFIT"
        
        pnl_percent = ((current_price / active_trade['entry_price']) - 1) * 100
        outcome = "PROFIT" if pnl_percent > 0 else "LOSS"
        
        if outcome == "PROFIT":
            trade_history['winning_trades'] += 1
        else:
            trade_history['losing_trades'] += 1
        
        # Update signal history
        for signal in trade_history['all_signals']:
            if signal.get('trade_id') == trade_id and signal.get('status') == 'OPEN':
                signal.update({
                    'exit_type': exit_type,
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'pnl_percent': pnl_percent,
                    'outcome': outcome,
                    'status': 'CLOSED'
                })
                break
        
        original_message_id = trade_history['message_ids'].get(trade_id)
        
        message = f"📉 <b>{exit_type} HIT</b> 📉\n\n"
        message += f"<b>Symbol:</b> {SYMBOL}\n"
        message += f"<b>Exit Price:</b> ${current_price:.2f}\n"
        message += f"<b>Entry Price:</b> ${active_trade['entry_price']:.2f}\n"
        message += f"<b>P/L:</b> {pnl_percent:.2f}%\n"
        message += f"<b>Outcome:</b> {outcome}\n"
        message += f"<b>Time:</b> {current_time}\n"
        message += f"<b>Updated Accuracy:</b> {calculate_accuracy():.2f}%"
        message += f"\n<b>Total Trades:</b> {trade_history['total_trades']}"
        message += f"\n<b>Win Rate:</b> {trade_history['winning_trades']}/{trade_history['total_trades']}"
        
        send_telegram_message(message, original_message_id)
        trade_history['active_trade'] = None
        
        save_signals_history()

def print_signal_summary():
    """Print summary of all signals"""
    if not trade_history['all_signals']:
        print("No signals in history")
        return
    
    print(f"\n=== SIGNAL HISTORY SUMMARY ===")
    print(f"Total Signals: {trade_history['total_trades']}")
    print(f"Winning Trades: {trade_history['winning_trades']}")
    print(f"Losing Trades: {trade_history['losing_trades']}")
    print(f"Accuracy: {calculate_accuracy():.2f}%")
    
    print(f"\n=== LAST 10 SIGNALS ===")
    for signal in trade_history['all_signals'][-10:]:
        if signal['type'] == 'BUY':
            print(f"BUY - {signal['timestamp']} - Price: ${signal['price']:.2f} - Status: {signal.get('status', 'OPEN')}")
        else:
            print(f"SELL - {signal['timestamp']} - Price: ${signal['price']:.2f}")

def main():
    """Main function to run the signal generator"""
    global trade_history
    
    # Load previous signals history
    trade_history = load_signals_history()
    print_signal_summary()
    
    # Initialize exchange
    try:
        exchange = getattr(ccxt, EXCHANGE_NAME)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        logging.info(f"Connected to {EXCHANGE_NAME} exchange")
    except Exception as e:
        logging.error(f"Failed to initialize exchange: {e}")
        return
    
    # Test data fetch
    logging.info("Testing data fetch...")
    test_df = fetch_ohlcv_data(exchange, SYMBOL, TIMEFRAME, 300)
    if test_df.empty:
        logging.error("Failed to fetch initial data. Check internet connection and exchange availability.")
        return
    
    # Send startup message
    startup_msg = "🤖 <b>Live Signal Generator Activated</b> 🤖\n\n"
    startup_msg += f"<b>Strategy:</b> Improved Trend Follower\n"
    startup_msg += f"<b>Symbol:</b> {SYMBOL}\n"
    startup_msg += f"<b>Timeframe:</b> {TIMEFRAME}\n"
    startup_msg += f"<b>Latest Price:</b> ${test_df['close'].iloc[-1]:.2f}\n"
    startup_msg += f"<b>Historical Accuracy:</b> {calculate_accuracy():.2f}%\n"
    startup_msg += f"<b>Total Signals:</b> {trade_history['total_trades']}\n"
    startup_msg += f"<b>Win Rate:</b> {trade_history['winning_trades']}/{trade_history['total_trades']}\n"
    startup_msg += "<i>Monitoring market for signals every 5 minutes...</i>"
    send_telegram_message(startup_msg)
    
    logging.info("Live Signal Generator Started...")
    logging.info(f"Historical Accuracy: {calculate_accuracy():.2f}%")
    logging.info(f"Total Signals: {trade_history['total_trades']}")
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            logging.info(f"\n=== CHECK #{check_count} - {datetime.datetime.now()} ===")
            
            # Fetch latest data
            df = fetch_ohlcv_data(exchange, SYMBOL, TIMEFRAME, 300)
            
            if not df.empty:
                logging.info("Checking for trading signals...")
                signals = check_trading_signals(df)
                
                if any([signals['buy_signal'], signals['sell_signal']]):
                    logging.info("Processing signal...")
                    process_signal(signals, df)
                else:
                    logging.info("No signals detected this cycle")
                    # Log market status every 6th check (every 30 minutes)
                    if check_count % 6 == 0:
                        logging.info(f"Market Status - Price: ${signals['current_price']:.2f}, "
                                    f"Fast MA: {signals['fast_ma']:.2f}, Slow MA: {signals['slow_ma']:.2f}")
            else:
                logging.warning("Empty dataframe received")
            
            # Show countdown for next check
            logging.info(f"Next check in 5 minutes...")
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logging.info("\nStopping signal generator...")
        save_signals_history()
        
        shutdown_msg = "🛑 <b>Signal Generator Stopped</b> 🛑\n\n"
        shutdown_msg += f"<b>Total Signals:</b> {trade_history['total_trades']}\n"
        shutdown_msg += f"<b>Final Accuracy:</b> {calculate_accuracy():.2f}%\n"
        shutdown_msg += f"<b>Win Rate:</b> {trade_history['winning_trades']}/{trade_history['total_trades']}"
        send_telegram_message(shutdown_msg)
        
    except Exception as e:
        error_msg = f"❌ <b>Signal Generator Error</b> ❌\n\nError: {str(e)}"
        send_telegram_message(error_msg)
        logging.error(f"Unexpected error: {e}")
        save_signals_history()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()