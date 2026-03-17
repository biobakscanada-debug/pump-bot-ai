import os
import time
import io
import threading
from datetime import datetime
import requests

import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from telegram import Bot
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Pump Hunter (стабильные публичные пары) is running!"

@app.route('/ping')
def ping():
    return "pong"

# ────────────────────────────────────────────────
# Константы — агрессивные для теста
# ────────────────────────────────────────────────
TIMEFRAME = '1h'
INTERVAL_SECONDS = 600
MODEL_FILE = 'catboost_pump_stable.cbm'
LAST_INDEX_FILE = 'last_pair_index.txt'

MIN_DATA_LENGTH = 60
PROBABILITY_THRESHOLD = 0.35
HIGH_PROB_NOTIFY_THRESHOLD = 0.40
SIGNAL_LIFETIME = 14400

VOLUME_SURGE = 1.2
PRICE_BREAK = 0.005
RSI_MIN = 40
RSI_MAX = 90

FEATURES = ['ema200', 'rsi', 'macd', 'bb_width', 'price_change', 'volume_change', 'volume_ratio']

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)

# ПУБЛИЧНЫЙ обменник — БЕЗ КЛЮЧЕЙ, для рынков, OHLCV, тикеров
public_exchange = ccxt.mexc({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# Приватный — ТОЛЬКО для фандинга
private_exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'adjustForTimeDifference': True, 'recvWindow': 15000}
})

PAIRS = []
ACTIVE_SIGNALS = []


# ────────────────────────────────────────────────
# Данные и фичи
# ────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, limit: int = 1500):
    try:
        time.sleep(0.85)
        bars = public_exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_DATA_LENGTH:
        return df

    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_width'] = (sma20 + std20*2 - (sma20 - std20*2)) / df['close']

    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(25).mean()

    return df.dropna()


# ────────────────────────────────────────────────
# Модель — 40 свежих пампов
# ────────────────────────────────────────────────
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    print("Обучение на 40 свежих пампах...")
    training_pairs = [
        'PEPE/USDT:USDT', 'WIF/USDT:USDT', 'BONK/USDT:USDT', 'POPCAT/USDT:USDT', 'BRETT/USDT:USDT',
        'FARTCOIN/USDT:USDT', 'GOAT/USDT:USDT', 'MOODENG/USDT:USDT', 'NEIRO/USDT:USDT', 'TRUMP/USDT:USDT',
        'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'SHIB/USDT:USDT', 'FLOKI/USDT:USDT', '1000BONK/USDT:USDT',
        'MEW/USDT:USDT', 'MOG/USDT:USDT', 'GIGA/USDT:USDT', 'PNUT/USDT:USDT', 'ACT/USDT:USDT',
        'TURBO/USDT:USDT', 'MIGGLES/USDT:USDT', 'TOSHI/USDT:USDT', 'BOME/USDT:USDT', 'SLERF/USDT:USDT',
        'FWOG/USDT:USDT', 'RETARDIO/USDT:USDT', 'LOCKIN/USDT:USDT', 'MOTHER/USDT:USDT', 'AURA/USDT:USDT',
        'DEGEN/USDT:USDT', 'HIGHER/USDT:USDT', 'BOBO/USDT:USDT', 'MUMU/USDT:USDT', 'KENDU/USDT:USDT',
        'CHEEMS/USDT:USDT', 'SAMO/USDT:USDT', 'KOKO/USDT:USDT', 'SELFIE/USDT:USDT', 'BILLY/USDT:USDT'
    ]

    all_data = []
    for symbol in training_pairs:
        try:
            time.sleep(2)
            df = fetch_ohlcv(symbol)
            if df.empty: continue
            df = add_features(df)
            if df.empty: continue
            df['target'] = (df['price_change'].shift(-1) > 0.018).astype(int)
            all_data.append(df)
        except Exception as e:
            print(f"Ошибка обучения на {symbol}: {e}")
            continue

    if not all_data:
        raise ValueError("Нет данных для обучения!")

    df_all = pd.concat(all_data).dropna()
    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=1200, depth=8, learning_rate=0.04, verbose=0)
    model.fit(X_tr, y_tr)
    model.save_model(MODEL_FILE)
    return model


def get_funding_rate(symbol):
    try:
        funding = private_exchange.fetch_funding_rate(symbol)
        return funding.get('fundingRate', 0) * 100
    except Exception as e:
        print(f"Funding ошибка для {symbol}: {e}")
        return 0.0


def send_funding_update(pair, funding_rate):
    sign = "📈" if funding_rate > 0 else "📉"
    text = f"📊 Фандинг {pair}\n{sign} {funding_rate:.4f}%\n"
    if funding_rate > 0.015:
        text += "⚠️ Вы платите за финансирование — может сожрать прибыль!"
    elif funding_rate < -0.01:
        text += "✅ Вам капает фандинг — держим позицию!"
    else:
        text += "Фандинг нейтральный."
    try:
        bot.send_message(CHAT_ID, text)
    except:
        pass


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return
    row = df.iloc[-1]

    if row['volume_ratio'] < VOLUME_SURGE or row['price_change'] < PRICE_BREAK or not (RSI_MIN < row['rsi'] < RSI_MAX):
        print(f"  Пропуск {pair} (prob={prob:.4f}) — не подтверждён памп")
        return

    text = f"""🟢 {pair.split('/')[0]} 🚀 Подтверждённый памп!
prob = {prob:.4f} | цена = {price:.8f} | объём x{row['volume_ratio']:.1f}
RSI = {row['rsi']:.1f} | импульс = {change*100:.2f}%

LONG на MEXC Futures
Цель 1: {round(price * 1.08, 8):.8f}
Цель 2: {round(price * 1.15, 8):.8f}
Стоп: {round(price * 0.94, 8):.8f} (-6%)"""

    buf = create_chart(pair, price)

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"🚀 СИГНАЛ ОТПРАВЛЕН → {pair}")
        ACTIVE_SIGNALS.append({'pair': pair, 'entry_price': price, 'timestamp': time.time()})
    except Exception as e:
        print(f"Ошибка отправки {pair}: {e}")


def create_chart(pair: str, entry_price: float):
    df = fetch_ohlcv(pair)
    if df.empty: return None
    df = add_features(df)
    if df.empty: return None

    tp1 = round(entry_price * 1.08, 6)
    tp2 = round(entry_price * 1.15, 6)
    stop = round(entry_price * 0.94, 6)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
    ax.plot(df['timestamp'], df['close'], color='#00ff9d', linewidth=2)
    ax.axhline(entry_price, color='white', linestyle='--', label='Вход')
    ax.axhline(tp1, color='#00ff00', label='Цель 1')
    ax.axhline(tp2, color='#00cc00', label='Цель 2')
    ax.axhline(stop, color='red', linestyle='--', label='Стоп')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.15)
    ax.set_title(f'{pair} — ПАМП', color='white')
    ax.legend(loc='upper left')
    ax.tick_params(colors='white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf


def update_pairs_list():
    global PAIRS
    try:
        markets = public_exchange.load_markets(reload=True)
        futures_pairs = [s for s, m in markets.items() if m.get('swap') and 'USDT' in s and m.get('active')]
        PAIRS[:] = sorted(futures_pairs, key=lambda s: float(markets[s].get('info', {}).get('quoteVolume', 0) or 0), reverse=True)
        print(f"Обновлён список: {len(PAIRS)} пар")
    except Exception as e:
        print(f"Ошибка обновления пар: {e}")
        # Если совсем упало — не крашим бот, оставляем старый список
        if len(PAIRS) == 0:
            print("Список пар пуст — ждём следующей итерации")


def load_last_index():
    if os.path.exists(LAST_INDEX_FILE):
        try:
            with open(LAST_INDEX_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def save_last_index(idx):
    try:
        with open(LAST_INDEX_FILE, 'w') as f:
            f.write(str(idx))
    except Exception as e:
        print(f"Ошибка сохранения индекса: {e}")


def check_expired_signals():
    global ACTIVE_SIGNALS
    now = time.time()
    to_remove = []
    for s in ACTIVE_SIGNALS:
        if now - s['timestamp'] > SIGNAL_LIFETIME:
            try:
                price, _, _ = get_market_data(s['pair'])
                msg = f"✅ {s['pair']} отработал!" if price > s['entry_price'] else f"⚠️ {s['pair']} тайм-аут"
                bot.send_message(CHAT_ID, msg)
            except:
                pass
            to_remove.append(s)
    ACTIVE_SIGNALS = [s for s in ACTIVE_SIGNALS if s not in to_remove]


def get_market_data(symbol):
    try:
        ticker = public_exchange.fetch_ticker(symbol)
        return ticker['last'], ticker.get('percentage', 0), round(ticker.get('quoteVolume', 0) / 1_000_000, 1)
    except:
        return 0.0, 0.0, 0.0


def main_loop():
    model = load_or_train_model()

    bot.send_message(CHAT_ID, f"🚀 Pump Hunter запущен (публичные пары без ошибок) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0
    last_funding_check = time.time()

    while True:
        iteration += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Итерация {iteration} | пар: {len(PAIRS)}")

        update_pairs_list()
        check_expired_signals()

        start_idx = load_last_index()
        print(f"[{now_str}] Продолжаем с индекса {start_idx}")

        scanned = 0
        high_prob_count = 0
        prob_list = []

        for i, pair in enumerate(PAIRS[start_idx:]):
            scanned += 1
            try:
                df = fetch_ohlcv(pair)
                if len(df) < MIN_DATA_LENGTH:
                    print(f"  {pair:20} → мало данных")
                    continue
                df = add_features(df)
                if df.empty:
                    print(f"  {pair:20} → фичи не посчитались")
                    continue

                row = df.iloc[-1]
                feats = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(feats)[0][1]

                print(f"  {pair:20} → prob={prob:.4f} | RSI={row['rsi']:.1f} | v_ratio={row['volume_ratio']:.1f}")

                if prob > HIGH_PROB_NOTIFY_THRESHOLD:
                    high_prob_count += 1
                    msg = f"🔥 Высокая вероятность (без фильтра): {pair}\nprob = {prob:.4f}\nRSI = {row['rsi']:.1f}\nv_ratio = {row['volume_ratio']:.1f}"
                    try:
                        bot.send_message(CHAT_ID, msg)
                        print(f"  Уведомление отправлено: {pair}")
                    except Exception as e:
                        print(f"  Ошибка уведомления {pair}: {e}")

                prob_list.append((pair, prob, row['rsi'], row['volume_ratio']))

                if prob > PROBABILITY_THRESHOLD:
                    price, _, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, row['price_change'])

            except Exception as e:
                print(f"  {pair} → ошибка: {type(e).__name__}")

            if scanned % 50 == 0:
                print(f"  Прогресс: обработано {scanned} пар из {len(PAIRS)} | последняя {pair}")

            current_idx = start_idx + i + 1
            save_last_index(current_idx)

            time.sleep(0.85)

        # Топ-5 каждые 3 итерации
        if prob_list and iteration % 3 == 0:
            top5 = sorted(prob_list, key=lambda x: x[1], reverse=True)[:5]
            top_text = f"Топ-5 вероятностей за итерацию {iteration}:\n"
            for pair, prob, rsi, vratio in top5:
                top_text += f"{pair}: prob={prob:.4f} | RSI={rsi:.1f} | v_ratio={vratio:.1f}\n"
            try:
                bot.send_message(CHAT_ID, top_text)
                print("Топ-5 отправлен в Telegram")
            except:
                print("Ошибка отправки топ-5")

        print(f"[{now_str}] Итерация завершена | просканировано {scanned} | уведомлений: {high_prob_count}")

        if time.time() - last_funding_check > 1800:
            for s in ACTIVE_SIGNALS[:]:
                try:
                    funding = get_funding_rate(s['pair'])
                    send_funding_update(s['pair'], funding)
                except:
                    pass
            last_funding_check = time.time()

        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
