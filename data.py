import websocket
import json

from snapshot import OrderBookSnapshot

def on_message(ws, message):
    try:
        print(message)
        data = json.loads(message)
        snapshot = OrderBookSnapshot.from_dict(data)
        
        # 이제 snapshot 객체를 사용하여 필요한 작업을 수행하십시오.
        print(f"Received snapshot for symbol: {snapshot.symbol}")
        print(f"First ask price: {snapshot.asks[0].price}, quantity: {snapshot.asks[0].quantity}")
        print(f"First bid price: {snapshot.bids[0].price}, quantity: {snapshot.bids[0].quantity}")
    except Exception as e:
        print(f"Error processing message: {e}")


def on_error(ws, error):
    print(f"Error occurred: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    message = {
        "type": "orderbooksnapshot",
        "symbols": ["BTC_KRW"]
    }
    ws.send(json.dumps(message))

if __name__ == "__main__":
    ws_url = "wss://pubwss.bithumb.com/pub/ws"
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
