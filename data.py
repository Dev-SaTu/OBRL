import csv
import numpy as np
import websocket
import json

from typing import List, Tuple

all_data = []

class Order:
    def __init__(self, price: str, quantity: str):
        self.price = price
        self.quantity = quantity

    @classmethod
    def from_tuple(cls, data: Tuple[str, str]):
        return cls(data[0], data[1])

class OrderBookSnapshot:
    def __init__(self, type_: str, symbol: str, datetime: str, asks: List[Order], bids: List[Order]):
        self.type = type_
        self.symbol = symbol
        self.datetime = datetime
        self.asks = asks
        self.bids = bids

    @classmethod
    def from_dict(cls, data: dict):
        asks = [Order.from_tuple(item) for item in data['content']['asks']]
        bids = [Order.from_tuple(item) for item in data['content']['bids']]
        return cls(data['type'], data['content']['symbol'], data['content']['datetime'], asks, bids)

def to_np_array(order_book: OrderBookSnapshot) -> np.ndarray:
    ask_data = []
    bid_data = []

    for order in order_book.asks:
        ask_data.extend([float(order.price), float(order.quantity)])

    for order in order_book.bids:
        bid_data.extend([float(order.price), float(order.quantity)])

    full_data = ask_data + bid_data
    full_array = np.array(full_data)
    
    return full_array

def on_message(ws, message):
    global all_data
    try:
        data = json.loads(message)
        snapshot = OrderBookSnapshot.from_dict(data)
        nparray = to_np_array(snapshot)
        all_data.append(nparray)
        
        print('len of all data: ', len(all_data))

        if len(all_data) >= 1000:
            with open('data.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row in all_data:
                    csvwriter.writerow(row)
            ws.close()
    except Exception as e:
        print(f"Error: {e}")

def on_open(ws):

    message = {
        "type": "orderbooksnapshot",
        "symbols": ["BTC_KRW"]
    }
    ws.send(json.dumps(message))

if __name__ == "__main__":

    ws_url = "wss://pubwss.bithumb.com/pub/ws"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
    ws.run_forever()
