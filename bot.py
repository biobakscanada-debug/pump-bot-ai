import os
import time
import io
import threading
from datetime import datetime, timedelta
import csv

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
    return "🚀 Pump Hunter — реальный инструмент!"

@app.route('/ping')
def ping():
    return "pong"

# Константы
TIMEFRAME = '1h'
MODEL_FILE = 'catboost_pump_real.cbm'
LAST_INDEX_FILE = 'last_pair_index.txt'
DATASET_FILE = 'pump_dataset.csv'
SIGNALS_LOG = 'signals_log.csv'

MIN_DATA_LENGTH = 60
PROBABILITY_THRESHOLD = 0.35
HIGH_PROB_NOTIFY_THRESHOLD = 0.40
SIGNAL_LIFETIME = 14400  # 4 часа

VOLUME_SURGE = 1.2
PRICE_BREAK = 0.005
RSI_MIN = 40
RSI_MAX = 90
ATR_MULTIPLIER = 3.0  # твой выбор

TP1_LEVEL = 1.08  # +8%
TP2_LEVEL = 1.15  # +15%
TRAIL_AFTER_TP1 = 1.03  # +3% после TP1

FEATURES = ['ema200', 'rsi', 'macd', 'bb_width', 'price_change', 'volume_change', 'volume_ratio']

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)

public_exchange = ccxt.mexc({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

private_exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'adjustForTimeDifference': True, 'recvWindow': 15000}
})

PAIRS = []
ACTIVE_SIGNALS = []  # [{'pair': 'PEPEUSDT', 'entry_price': 0.00001, 'timestamp': time.time(), 'atr': 0.000001, 'status': 'open'}]


def fetch_ohlcv(symbol: str, limit: int = 1500):
    try:
        time.sleep(0.45)
        bars = public_exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.RateLimitExceeded:
        print(f"Rate limit на {symbol}, ждём 5 сек")
        time.sleep(5)
        return fetch_ohlcv(symbol, limit)
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

    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift()), 
                                     abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()

    return df.dropna()


def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        print("Загружаем модель...")
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    if not os.path.exists(DATASET_FILE):
        print(f"Файл {DATASET_FILE} не найден! Обучение невозможно.")
        return None

    print(f"Загружаем датасет из {DATASET_FILE}...")
    df_all = pd.read_csv(DATASET_FILE)
    if df_all.empty:
        print("Датасет пуст!")
        return None

    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=1200, depth=8, learning_rate=0.04, verbose=0)
    model.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"Модель обучена на реальном датасете | Accuracy: {acc:.4f} ({acc*100:.2f}%) | Строк: {len(df_all)}")

    model.save_model(MODEL_FILE)
    return model


def log_signal(signal_data):
    file_exists = os.path.exists(SIGNALS_LOG)
    with open(SIGNALS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=signal_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(signal_data)


def check_signals_status():
    now = time.time()
    to_remove = []
    report = []

    for s in ACTIVE_SIGNALS:
        pair = s['pair']
        entry = s['entry_price']
        atr = s['atr']
        status = s.get('status', 'open')
        tp1_hit = s.get('tp1_hit', False)

        try:
            price, _, _ = get_market_data(pair)
            if price <= 0:
                continue

            # TP1
            if not tp1_hit and price >= entry * TP1_LEVEL:
                s['tp1_hit'] = True
                s['trail_sl'] = entry * TRAIL_AFTER_TP1
                report.append(f"TP1 достигнут: {pair} | {price:.8f}")

            # TP2
            if price >= entry * TP2_LEVEL:
                report.append(f"TP2 достигнут: {pair} | {price:.8f}")
                to_remove.append(s)
                continue

            # SL или trailing SL
            current_sl = s['trail_sl'] if tp1_hit else (entry - atr * ATR_MULTIPLIER)
            if price <= current_sl:
                report.append(f"SL сработал: {pair} | {price:.8f}")
                to_remove.append(s)
                continue

            # Время жизни сигнала
            if now - s['timestamp'] > SIGNAL_LIFETIME:
                report.append(f"Тайм-аут: {pair} | {price:.8f}")
                to_remove.append(s)
                continue

        except Exception as e:
            print(f"Ошибка проверки {pair}: {e}")

    for s in to_remove:
        ACTIVE_SIGNALS.remove(s)

    if report:
        bot.send_message(CHAT_ID, "\n".join(report))


def daily_report():
    global last_report_time
    now = time.time()
    if now - last_report_time < 86400:  # 24 часа
        return

    if not os.path.exists(SIGNALS_LOG):
        return

    df = pd.read_csv(SIGNALS_LOG)
    if df.empty:
        return

    # Простая статистика (можно расширить)
    total = len(df)
    tp_hit = len(df[df['status'] == 'tp_hit'])
    sl_hit = len(df[df['status'] == 'sl_hit'])
    timeout = len(df[df['status'] == 'timeout'])
    winrate = tp_hit / total * 100 if total > 0 else 0

    text = f"""📊 Ежедневный отчёт
Всего сигналов: {total}
TP достигнуто: {tp_hit} ({winrate:.1f}%)
SL сработал: {sl_hit}
Тайм-аут: {timeout}
Активных сигналов: {len(ACTIVE_SIGNALS)}"""

    bot.send_message(CHAT_ID, text)
    last_report_time = now


last_report_time = time.time()


def main_loop():
    model = load_or_train_model()
    if model is None:
        print("Модель не загружена")
    else:
        print("Модель готова")

    bot.send_message(CHAT_ID, f"🚀 Pump Hunter запущен — журнал в CSV | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0
    last_funding_check = time.time()

    while True:
        iteration += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Итерация {iteration} | пар: {len(PAIRS)}")

        update_pairs_list()
        check_expired_signals()
        check_signals_status()
        daily_report()

        if len(PAIRS) == 0:
            print("Пар нет, пробуем обновить...")
            update_pairs_list()
            time.sleep(60)
            continue

        start_idx = load_last_index()
        if start_idx >= len(PAIRS):
            start_idx = 0
            save_last_index(0)

        scanned = 0
        high_prob_count = 0
        prob_list = []

        for i, pair in enumerate(PAIRS[start_idx:]):
            scanned += 1
            try:
                df = fetch_ohlcv(pair)
                if len(df) < MIN_DATA_LENGTH:
                    continue
                df = add_features(df)
                if df.empty:
                    continue

                row = df.iloc[-1]
                feats = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(feats)[0][1] if model is not None else 0.0

                print(f"  {pair:20} → prob={prob:.4f} | RSI={row['rsi']:.1f} | v_ratio={row['volume_ratio']:.1f}")

                if prob > HIGH_PROB_NOTIFY_THRESHOLD:
                    high_prob_count += 1
                    msg = f"🔥 Высокая вероятность: {pair}\nprob = {prob:.4f}\nRSI = {row['rsi']:.1f}\nv_ratio = {row['volume_ratio']:.1f}"
                    bot.send_message(CHAT_ID, msg)

                prob_list.append((pair, prob, row['rsi'], row['volume_ratio']))

                if prob > PROBABILITY_THRESHOLD:
                    price, _, vm = get_market_data(pair)
                    atr = row['atr']
                    send_signal(pair, price, prob, vm, row['price_change'], atr)

                    # Логируем сигнал
                    log_signal({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'pair': pair,
                        'entry_price': price,
                        'prob': prob,
                        'rsi': row['rsi'],
                        'v_ratio': row['volume_ratio'],
                        'atr': atr,
                        'status': 'open'
                    })

            except Exception as e:
                print(f"  {pair} → ошибка: {type(e).__name__}")

            if scanned % 50 == 0:
                print(f"  Прогресс: {scanned}/{len(PAIRS)} | {pair}")

            current_idx = start_idx + i + 1
            save_last_index(current_idx)

            time.sleep(0.45)

        if prob_list and iteration % 3 == 0:
            top5 = sorted(prob_list, key=lambda x: x[1], reverse=True)[:5]
            top_text = f"Топ-5 за итерацию {iteration}:\n"
            for pair, prob, rsi, vratio in top5:
                top_text += f"{pair}: prob={prob:.4f} | RSI={rsi:.1f} | v_ratio={vratio:.1f}\n"
            bot.send_message(CHAT_ID, top_text)

        print(f"[{now_str}] Итерация завершена | просканировано {scanned} | уведомлений: {high_prob_count} → сразу следующая")

        if time.time() - last_funding_check > 1800:
            for s in ACTIVE_SIGNALS[:]:
                try:
                    funding = get_funding_rate(s['pair'])
                    send_funding_update(s['pair'], funding)
                except:
                    pass
            last_funding_check = time.time()


if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
