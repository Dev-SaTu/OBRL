from typing import List, Tuple

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