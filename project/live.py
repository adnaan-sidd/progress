import os
import asyncio
import logging
from datetime import datetime
import joblib
import pandas as pd
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi
from utils.features import calculate_technical_indicators, feature_engineering

# Configure logging with reduced verbosity
logging.basicConfig(
    level=logging.WARNING,  # Show only warnings and errors
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv('METAAPI_TOKEN')
ACCOUNT_ID = os.getenv('METAAPI_ACCOUNT_ID')
SYMBOL = 'EURUSD'
TRADING_HOURS = {'start': 0, 'end': 24}  # 24-hour trading

class ForexTradingBot:
    def __init__(self):
        self.client = MetaApi(API_TOKEN)
        self.account = None
        self.connection = None
        self.model = joblib.load('eurusd_model.pkl')
        self.equity = 34.0
        self.trailing_stops = {}
        self.trade_history = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }

    async def connect(self):
        retries = 3
        for attempt in range(retries):
            try:
                self.account = await self.client.metatrader_account_api.get_account(ACCOUNT_ID)
                self.connection = self.account.get_streaming_connection()  # Synchronous call
                await self.connection.connect()  # Async operation
                await self.connection.wait_synchronized()  # No timeout_in_seconds
                logger.info("Successfully connected to MetaApi")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(5)
                else:
                    raise Exception("Failed to connect to MetaApi after multiple attempts")

    async def ensure_connection(self):
        """Ensure the connection is active, reconnect if necessary."""
        try:
            if not self.connection:
                raise Exception("No connection established")
            await self.connection.wait_synchronized()  # No timeout_in_seconds
        except Exception as e:
            logger.warning(f"Connection issue detected: {str(e)}. Attempting to reconnect...")
            await self.connect()

    async def get_market_data(self):
        try:
            # Fetch 5-minute candles
            candles = await self.connection.get_historical_candles(SYMBOL, '5m', 50)
            df = pd.DataFrame(candles)
            return self.calculate_features(df)
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None

    def calculate_features(self, df):
        try:
            df = calculate_technical_indicators(df)
            return feature_engineering(df).iloc[-1]
        except Exception as e:
            logger.error(f"Feature calculation error: {str(e)}")
            return None

    async def trading_cycle(self):
        while self.equity > 30:
            try:
                await self.ensure_connection()

                if not self.within_trading_hours():
                    logger.info("Outside trading hours, sleeping...")
                    await asyncio.sleep(300)
                    continue

                features = await self.get_market_data()
                if features is None:
                    await asyncio.sleep(60)
                    continue

                prediction = self.model.predict([features])[0]
                price = (await self.connection.get_symbol_specification(SYMBOL))['bid']
                positions = await self.connection.get_positions()

                if not any(pos['symbol'] == SYMBOL for pos in positions):
                    await self.execute_trade(prediction, price)
                else:
                    await self.update_trailing_stop(price)

                account_info = await self.connection.get_account_information()
                self.equity = account_info['equity']
                logger.info(f"Current equity: ${self.equity:.2f}")
                self.log_performance()

                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Trading error: {str(e)}")
                await asyncio.sleep(60)

    async def execute_trade(self, prediction, price):
        try:
            lot_size = 0.01
            stop_loss = price * 0.9995 if prediction == 1 else price * 1.0005
            trade_type = 'buy' if prediction == 1 else 'sell'
            take_profit = price * 1.0015 if prediction == 1 else price * 0.9985

            order = await (self.connection.create_market_buy_order if prediction == 1 else
                         self.connection.create_market_sell_order)(
                SYMBOL, lot_size, stop_loss=stop_loss, take_profit=take_profit
            )

            trade_record = {
                'time': datetime.utcnow(),
                'type': trade_type,
                'price': price,
                'lot_size': lot_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': order['id']
            }
            self.trade_history.append(trade_record)
            self.performance['total_trades'] += 1

            logger.info(f"Opened {trade_type.upper()} position at {price:.5f}")
            self.trailing_stops[SYMBOL] = {
                'type': trade_type,
                'peak' if prediction == 1 else 'trough': price
            }

        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")

    async def update_trailing_stop(self, current_price):
        try:
            trail_info = self.trailing_stops.get(SYMBOL)
            if not trail_info:
                return

            if trail_info['type'] == 'buy' and current_price > trail_info['peak']:
                new_sl = current_price - 0.00015
                positions = await self.connection.get_positions()
                for pos in positions:
                    if pos['symbol'] == SYMBOL:
                        await self.connection.modify_position(pos['id'], stop_loss=new_sl)
                        trail_info['peak'] = current_price
                        logger.info(f"Updated BUY stop loss to {new_sl:.5f}")
                        break

            elif trail_info['type'] == 'sell' and current_price < trail_info['trough']:
                new_sl = current_price + 0.00015
                positions = await self.connection.get_positions()
                for pos in positions:
                    if pos['symbol'] == SYMBOL:
                        await self.connection.modify_position(pos['id'], stop_loss=new_sl)
                        trail_info['trough'] = current_price
                        logger.info(f"Updated SELL stop loss to {new_sl:.5f}")
                        break

        except Exception as e:
            logger.error(f"Trailing stop update failed: {str(e)}")

    def log_performance(self):
        """Log current performance metrics"""
        logger.info(f"Performance Metrics: "
                   f"Total Trades: {self.performance['total_trades']}, "
                   f"Win Rate: {self.performance['winning_trades']/self.performance['total_trades']*100:.2f}% "
                   f"if {self.performance['total_trades']} > 0 else 'N/A', "
                   f"Total PnL: ${self.performance['total_pnl']:.2f}")

    def update_performance(self, trade_result):
        """Update performance metrics after trade closure"""
        if trade_result['profit'] > 0:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        self.performance['total_pnl'] += trade_result['profit']

    def within_trading_hours(self):
        now = datetime.utcnow()
        return TRADING_HOURS['start'] <= now.hour < TRADING_HOURS['end']

    def save_trade_history(self):
        """Save trade history to CSV"""
        pd.DataFrame(self.trade_history).to_csv('trade_history.csv', index=False)
        logger.info("Trade history saved to trade_history.csv")

async def main():
    bot = ForexTradingBot()
    try:
        await bot.connect()
        await bot.trading_cycle()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, closing gracefully...")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
    finally:
        if bot.connection:
            await bot.connection.close()
        bot.save_trade_history()
        logger.info("Trading bot stopped")
        await asyncio.sleep(1)  # Allow time for tasks to complete

if __name__ == '__main__':
    asyncio.run(main())
