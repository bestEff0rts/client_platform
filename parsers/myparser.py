#myparser.py функции для импортов в ml скриптах: from parser import fetch_binance_ohlcv, fetch_moex_ohlcv
# parser.py будет конфликт
import ccxt
import aiohttp
import asyncio
import pandas as pd
import time

async def fetch_binance_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1d", since: int = None, limit: int = 500):
    exchange = ccxt.binance()
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(
            None, lambda: exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        )
        return data
    except Exception as e:
        return {"error": str(e)}

# ---------------------- FETCH MOEX ----------------------
# async def fetch_moex_ohlcv(ticker: str = "GAZP"): ИНТРАДЕЙ МИНУТНЫЕ ДАННЫЕ ТУТ
#     # Тестируем в разных рынках
#     markets = ["shares", "index", "futures"]
#     for market in markets:
#         url = f"https://iss.moex.com/iss/engines/stock/markets/{market}/securities/{ticker}/candles.json?interval=1"
#         async with aiohttp.ClientSession() as session:
#             async with session.get(url) as resp:
#                 if resp.status != 200:
#                     continue
#                 try:
#                     data = await resp.json()
#                     candles = data.get("candles", {}).get("data", [])
#                     columns = data.get("candles", {}).get("columns", [])
#                     if candles:
#                         df = pd.DataFrame(candles, columns=columns)
#                         return df.to_dict(orient="records")
#                 except Exception as e:
#                     continue
#     return {"error": f"MOEX: Instrument '{ticker}' not found in supported markets."}
import pandas as pd
import aiohttp
import asyncio
from datetime import datetime, timedelta

async def fetch_moex_ohlcv(ticker: str = "GAZP", interval: int = 1, days: int = 5):
    markets = ["shares", "index", "futures"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    for market in markets:
        url_base = f"https://iss.moex.com/iss/engines/stock/markets/{market}/securities/{ticker}/candles.json"
        all_data = []

        async with aiohttp.ClientSession() as session:
            start = 0
            while True:
                params = {
                    "interval": interval,
                    "from": start_str,
                    "till": end_str,
                    "start": start
                }
                async with session.get(url_base, params=params) as resp:
                    if resp.status != 200:
                        break
                    try:
                        data = await resp.json()
                        candles = data.get("candles", {}).get("data", [])
                        columns = data.get("candles", {}).get("columns", [])
                        if not candles:
                            break
                        all_data.extend(candles)
                        start += len(candles)
                        if len(candles) < 5000:
                            break  # достигнут конец
                    except Exception as e:
                        break

        if all_data:
            df = pd.DataFrame(all_data, columns=columns)
            df["begin"] = pd.to_datetime(df["begin"])
            df.set_index("begin", inplace=True)
            return df

    return pd.DataFrame(), f"MOEX: Instrument '{ticker}' not found in supported markets."
