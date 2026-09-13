"""
Knowledge Base Module
Represents structured facts about the UTH canteen menu items.
Each item is a frame/record with attributes (name, price, category, diet type, time availability).
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum


class MealTime(Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"


class DietType(Enum):
    MAN = "mặn"  # regular
    CHAY = "chay"  # vegetarian
    BOTH = "cả hai"  # suitable for both


class Category(Enum):
    COM = "cơm"
    MI_BUN_PHO = "mì/bún/phở"
    BANH_MI = "bánh mì"
    NUOC_EP = "nước ép"
    CA_PHE = "cà phê"
    TRA_SUA = "trà sữa"
    KHAC = "khác"


@dataclass
class MenuItem:
    """Frame/Record representation of a menu item in the knowledge base"""
    name: str
    price: float  # VND, use minimum price for ranges
    price_max: Optional[float]  # for items with price ranges
    category: Category
    diet: DietType
    meal_time: MealTime
    calories_estimate: int  # rough estimate for hunger matching

    def get_price_range(self) -> tuple:
        """Returns (min_price, max_price)"""
        return (self.price, self.price_max if self.price_max else self.price)

    def matches_diet(self, preferred_diet: DietType) -> bool:
        """Check if this item matches the dietary preference"""
        if self.diet == DietType.BOTH or preferred_diet == DietType.BOTH:
            return True
        return self.diet == preferred_diet


class KnowledgeBase:
    """
    Central knowledge repository for all canteen menu items.
    Implements structured knowledge representation using frames/records.
    """

    def __init__(self):
        self.dishes: List[MenuItem] = []
        self.drinks: List[MenuItem] = []
        self._initialize_knowledge()

    def _initialize_knowledge(self):
        """Load all menu items into structured knowledge base"""
        self._load_breakfast_dishes()
        self._load_lunch_dishes()
        self._load_drinks()

    def _categorize_by_name(self, name: str) -> Category:
        """Infer category from item name using keyword matching"""
        name_lower = name.lower()

        if any(kw in name_lower for kw in ["cơm", "rice"]):
            return Category.COM
        elif "bánh mì" in name_lower:
            return Category.BANH_MI
        elif any(kw in name_lower for kw in ["mì", "bún", "phở", "hủ tiếu", "bánh canh", "nui", "hảo hảo"]):
            return Category.MI_BUN_PHO
        elif any(kw in name_lower for kw in ["táo", "bưởi", "cà rốt", "thơm", "chanh", "dưa", "cam", "cóc", "ổi"]):
            return Category.NUOC_EP
        elif any(kw in name_lower for kw in ["café", "cafe", "coffee", "cà phê", "bạc xỉu"]):
            return Category.CA_PHE
        elif "trà" in name_lower:
            return Category.TRA_SUA
        else:
            return Category.KHAC

    def _estimate_calories(self, name: str, price: float, category: Category) -> int:
        """Rough calorie estimation for hunger level matching"""
        # Simple heuristic based on dish type and price
        if category == Category.COM:
            return int(500 + (price / 1000) * 10)
        elif category == Category.MI_BUN_PHO:
            return int(400 + (price / 1000) * 8)
        elif category == Category.BANH_MI:
            return int(300 + (price / 1000) * 5)
        else:
            return 200

    def _load_breakfast_dishes(self):
        """Load breakfast menu items"""
        breakfast_items = [
            {"name": "Bánh mì ốp la", "price": 17000},
            {"name": "Bánh mì xá xíu", "price": 17000},
            {"name": "Bánh mì phá lấu", "price": 17000},
            {"name": "Hảo hảo ốp la", "price": 17000},
            {"name": "Bánh ướt", "price": 22000, "price_max": 27000},
            {"name": "Mì trộn sa tế ốp la", "price": 27000},
            {"name": "Cơm chiên ốp la", "price": 27000},
            {"name": "Nui/Mì xào ốp la", "price": 27000},
            {"name": "Nui/Mì xào bò", "price": 35000},
            {"name": "Cơm chiên gà xối mỡ", "price": 35000},
            {"name": "Cơm chiên bò/xá xíu", "price": 35000},
            {"name": "Bún bò/Phở gà", "price": 35000},
            {"name": "Hủ tiếu/Nui/Bánh canh", "price": 35000}
        ]

        for item in breakfast_items:
            category = self._categorize_by_name(item["name"])
            price = item["price"]
            calories = self._estimate_calories(item["name"], price, category)

            # Most breakfast items are regular (mặn), but some can be made vegetarian
            diet = DietType.BOTH if "chay" in item["name"].lower() else DietType.MAN

            menu_item = MenuItem(
                name=item["name"],
                price=price,
                price_max=item.get("price_max"),
                category=category,
                diet=diet,
                meal_time=MealTime.BREAKFAST,
                calories_estimate=calories
            )
            self.dishes.append(menu_item)

    def _load_lunch_dishes(self):
        """Load lunch menu items (student rice plates)"""
        lunch_items = [
            {"name": "Cơm chay", "price": 10000, "diet": "chay"},
            {"name": "Cơm phần 27k", "price": 27000, "diet": "man"},
            {"name": "Cơm phần 31k", "price": 31000, "diet": "man"},
            {"name": "Cơm phần 35k", "price": 35000, "diet": "man"},
            {"name": "Cơm 2 món", "price": 45000, "diet": "man"}
        ]

        for item in lunch_items:
            category = Category.COM
            price = item["price"]
            calories = self._estimate_calories(item["name"], price, category)
            diet = DietType.CHAY if item["diet"] == "chay" else DietType.MAN

            menu_item = MenuItem(
                name=item["name"],
                price=price,
                price_max=None,
                category=category,
                diet=diet,
                meal_time=MealTime.LUNCH,
                calories_estimate=calories
            )
            self.dishes.append(menu_item)

    def _load_drinks(self):
        """Load all drink items"""
        # Juice items
        juice_items = [
            {"name": "Nước ép táo/bưởi", "price": 30000, "price_max": 35000},
            {"name": "Nước ép cà rốt", "price": 25000, "price_max": 30000},
            {"name": "Nước ép thơm", "price": 25000, "price_max": 30000},
            {"name": "Nước ép chanh dây", "price": 25000, "price_max": 30000},
            {"name": "Nước ép dưa hấu", "price": 20000, "price_max": 25000},
            {"name": "Nước ép cam", "price": 20000, "price_max": 25000},
            {"name": "Nước ép cóc", "price": 20000, "price_max": 25000},
            {"name": "Nước ép ổi", "price": 20000, "price_max": 25000}
        ]

        for item in juice_items:
            self.drinks.append(MenuItem(
                name=item["name"],
                price=item["price"],
                price_max=item.get("price_max"),
                category=Category.NUOC_EP,
                diet=DietType.BOTH,
                meal_time=MealTime.BREAKFAST,  # available all day
                calories_estimate=100
            ))

        # Coffee items
        coffee_items = [
            {"name": "Café đen đá", "price": 17000},
            {"name": "Café sữa đá", "price": 20000},
            {"name": "Bạc xỉu", "price": 22000},
            {"name": "Sữa tươi café", "price": 25000}
        ]

        for item in coffee_items:
            self.drinks.append(MenuItem(
                name=item["name"],
                price=item["price"],
                price_max=None,
                category=Category.CA_PHE,
                diet=DietType.BOTH,
                meal_time=MealTime.BREAKFAST,
                calories_estimate=50
            ))

        # Tea items
        tea_items = [
            {"name": "Trà tắc/đào tắc/vải tắc", "price": 17000},
            {"name": "Trà sữa truyền thống", "price": 18000, "price_max": 22000},
            {"name": "Trà sữa thái xanh", "price": 18000, "price_max": 22000},
            {"name": "Trà me", "price": 20000}
        ]

        for item in tea_items:
            self.drinks.append(MenuItem(
                name=item["name"],
                price=item["price"],
                price_max=item.get("price_max"),
                category=Category.TRA_SUA if "trà sữa" in item["name"].lower() else Category.KHAC,
                diet=DietType.BOTH,
                meal_time=MealTime.BREAKFAST,
                calories_estimate=80
            ))

        # Other drinks
        other_drinks = [
            {"name": "Nước suối", "price": 7000},
            {"name": "Nước ngọt/Milo/Sữa hộp", "price": 13000},
            {"name": "Highlands cafe lon", "price": 15000},
            {"name": "Redbull", "price": 17000},
            {"name": "Betagen/YoGood", "price": 25000},
            {"name": "Rau má nguyên chất", "price": 12000, "price_max": 22000},
            {"name": "Rau má sữa", "price": 18000, "price_max": 28000},
            {"name": "Sữa đậu nành", "price": 15000, "price_max": 23000},
            {"name": "Sữa đậu nành sương sáo", "price": 20000, "price_max": 28000},
            {"name": "Nước sâm", "price": 17000}
        ]

        for item in other_drinks:
            self.drinks.append(MenuItem(
                name=item["name"],
                price=item["price"],
                price_max=item.get("price_max"),
                category=Category.KHAC,
                diet=DietType.BOTH,
                meal_time=MealTime.BREAKFAST,
                calories_estimate=60
            ))

    def get_all_dishes(self) -> List[MenuItem]:
        """Retrieve all dishes from knowledge base"""
        return self.dishes.copy()

    def get_all_drinks(self) -> List[MenuItem]:
        """Retrieve all drinks from knowledge base"""
        return self.drinks.copy()

    def query_dishes(self, meal_time: Optional[MealTime] = None,
                     diet: Optional[DietType] = None,
                     category: Optional[Category] = None) -> List[MenuItem]:
        """Query dishes with optional filters"""
        results = self.dishes.copy()

        if meal_time:
            results = [d for d in results if d.meal_time == meal_time]

        if diet:
            results = [d for d in results if d.matches_diet(diet)]

        if category:
            results = [d for d in results if d.category == category]

        return results

    def query_drinks(self, category: Optional[Category] = None) -> List[MenuItem]:
        """Query drinks with optional category filter"""
        results = self.drinks.copy()

        if category:
            results = [d for d in results if d.category == category]

        return results
