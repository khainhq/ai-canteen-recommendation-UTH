"""
Constraint Satisfaction Problem (CSP) Module
Implements the meal recommendation problem as a formal CSP with variables, domains, and constraints.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from knowledge_base import MenuItem, MealTime, DietType, Category


class CSPVariable(Enum):
    """Variables in the meal recommendation CSP"""
    FOOD = "Food"
    DRINK = "Drink"


@dataclass
class UserRequest:
    """
    Formal representation of input X in f(X)
    X = (B, T, V, H, P, D)
    where:
        B = Budget (VND)
        T = Meal time (breakfast/lunch)
        V = Vegetarian requirement (true/false)
        H = Hunger level (light/normal/very_hungry)
        P = Food preferences (list of category preferences)
        D = Drink requirement (true/false)
    """
    budget: float  # B
    meal_time: MealTime  # T
    vegetarian: bool  # V
    hunger_level: str  # H in {light, normal, very_hungry}
    taste_preferences: List[str]  # P
    wants_drink: bool  # D

    # Additional soft preferences
    is_thirsty: bool = True
    caffeine_allowed: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            "Budget (B)": f"{int(self.budget):,} VND",
            "MealTime (T)": self.meal_time.value,
            "Vegetarian (V)": "Yes" if self.vegetarian else "No",
            "Hunger (H)": self.hunger_level,
            "Preferences (P)": ", ".join(self.taste_preferences) if self.taste_preferences else "None",
            "Wants Drink (D)": "Yes" if self.wants_drink else "No"
        }


@dataclass
class Assignment:
    """Represents a partial or complete variable assignment in CSP"""
    food: Optional[MenuItem] = None
    drink: Optional[MenuItem] = None

    def is_complete(self, wants_drink: bool) -> bool:
        """Check if assignment is complete based on problem requirements"""
        if wants_drink:
            return self.food is not None and self.drink is not None
        else:
            return self.food is not None

    def get_total_price(self) -> float:
        """Calculate total price of current assignment"""
        total = 0.0
        if self.food:
            total += self.food.price
        if self.drink:
            total += self.drink.price
        return total

    def copy(self) -> 'Assignment':
        """Create a copy of this assignment"""
        return Assignment(food=self.food, drink=self.drink)


class Constraint:
    """
    Base class for CSP constraints
    Each constraint can check if an assignment is consistent
    """

    def __init__(self, constraint_id: str, description: str, is_hard: bool = True):
        self.id = constraint_id
        self.description = description
        self.is_hard = is_hard  # Hard constraint vs soft preference

    def is_satisfied(self, assignment: Assignment, request: UserRequest) -> bool:
        """Check if this constraint is satisfied by the assignment"""
        raise NotImplementedError


class BudgetConstraint(Constraint):
    """C1: Budget constraint - total price must not exceed budget"""

    def __init__(self):
        super().__init__(
            "C1",
            "total_price ≤ budget",
            is_hard=True
        )

    def is_satisfied(self, assignment: Assignment, request: UserRequest) -> bool:
        total_price = assignment.get_total_price()
        return total_price <= request.budget


class MealTimeConstraint(Constraint):
    """C2: Meal time availability constraint"""

    def __init__(self):
        super().__init__(
            "C2",
            "meal items must be available at requested time",
            is_hard=True
        )

    def is_satisfied(self, assignment: Assignment, request: UserRequest) -> bool:
        if assignment.food:
            if assignment.food.meal_time != request.meal_time:
                return False
        # Drinks are available at all times
        return True


class VegetarianConstraint(Constraint):
    """C3: Vegetarian dietary constraint"""

    def __init__(self):
        super().__init__(
            "C3",
            "IF vegetarian=true THEN food must be vegetarian-compatible",
            is_hard=True
        )

    def is_satisfied(self, assignment: Assignment, request: UserRequest) -> bool:
        if not request.vegetarian:
            return True  # No restriction

        if assignment.food:
            diet_type = DietType.CHAY if request.vegetarian else DietType.MAN
            if not assignment.food.matches_diet(diet_type):
                return False

        return True


class CaffeineConstraint(Constraint):
    """C4: Caffeine restriction constraint"""

    def __init__(self):
        super().__init__(
            "C4",
            "IF caffeine_allowed=false THEN exclude caffeinated drinks",
            is_hard=True
        )

    def is_satisfied(self, assignment: Assignment, request: UserRequest) -> bool:
        if request.caffeine_allowed:
            return True  # No restriction

        if assignment.drink:
            # Coffee is caffeinated
            if assignment.drink.category == Category.CA_PHE:
                return False

        return True


class PrimaryItemConstraint(Constraint):
    """
    C5: Primary item constraint - add-ons cannot be the primary meal
    Add-ons include: extra egg, extra noodles, extra rice, toppings, surcharges
    """

    ADDON_KEYWORDS = [
        "thêm",  # extra
        "ốp la thêm",  # extra egg
        "vắt mì thêm",  # extra noodles
        "cơm thêm",  # extra rice
        "thức ăn thêm",  # extra food
        "mang về",  # take-away (surcharge)
        "trân châu",  # tapioca pearls (topping)
        "sương sáo",  # grass jelly (topping)
        "tẩy đá"  # ice
    ]

    def __init__(self):
        super().__init__(
            "C5",
            "add-ons cannot be the primary meal item",
            is_hard=True
        )

    def is_satisfied(self, assignment: Assignment, request: UserRequest) -> bool:
        if assignment.food:
            name_lower = assignment.food.name.lower()
            for keyword in self.ADDON_KEYWORDS:
                if keyword in name_lower and not any(
                    base in name_lower for base in ["bánh mì", "cơm", "mì", "bún", "phở"]
                ):
                    return False

        return True


class CSPModel:
    """
    Formal CSP representation of the meal recommendation problem

    Variables: {Food, Drink}
    Domains: D_Food = {food items}, D_Drink = {drink items}
    Constraints: {C1, C2, C3, C4, C5, ...}
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.constraints: List[Constraint] = []
        self._initialize_constraints()

    def _initialize_constraints(self):
        """Initialize all hard constraints"""
        self.constraints = [
            BudgetConstraint(),
            MealTimeConstraint(),
            VegetarianConstraint(),
            CaffeineConstraint(),
            PrimaryItemConstraint()
        ]

    def get_initial_domains(self, request: UserRequest) -> Dict[CSPVariable, List[MenuItem]]:
        """
        Get initial domains for CSP variables before any constraint propagation
        D_Food = all food items
        D_Drink = all drink items (or empty if drink not wanted)
        """
        domains = {
            CSPVariable.FOOD: self.kb.get_all_dishes(),
            CSPVariable.DRINK: self.kb.get_all_drinks() if request.wants_drink else []
        }
        return domains

    def propagate_constraints(self,
                             domains: Dict[CSPVariable, List[MenuItem]],
                             request: UserRequest) -> Tuple[Dict[CSPVariable, List[MenuItem]], Dict]:
        """
        Constraint propagation: reduce domains by applying unary constraints
        before backtracking search

        Returns: (reduced_domains, propagation_stats)
        """
        stats = {
            "initial_food_count": len(domains[CSPVariable.FOOD]),
            "initial_drink_count": len(domains[CSPVariable.DRINK]),
            "removed_by_meal_time": 0,
            "removed_by_diet": 0,
            "removed_by_budget": 0,
            "removed_by_caffeine": 0,
            "removed_by_addon": 0,
            "final_food_count": 0,
            "final_drink_count": 0
        }

        # Create temporary assignment for testing
        food_domain = domains[CSPVariable.FOOD].copy()
        drink_domain = domains[CSPVariable.DRINK].copy()

        # Filter food domain
        initial_food = len(food_domain)

        # Apply meal time constraint
        food_domain = [f for f in food_domain if f.meal_time == request.meal_time]
        stats["removed_by_meal_time"] = initial_food - len(food_domain)

        # Apply diet constraint
        before_diet = len(food_domain)
        if request.vegetarian:
            diet_type = DietType.CHAY
            food_domain = [f for f in food_domain if f.matches_diet(diet_type)]
        stats["removed_by_diet"] = before_diet - len(food_domain)

        # Apply budget constraint (single item must be affordable)
        before_budget = len(food_domain)
        food_domain = [f for f in food_domain if f.price <= request.budget]
        stats["removed_by_budget"] = before_budget - len(food_domain)

        # Apply add-on constraint
        before_addon = len(food_domain)
        food_domain = [
            f for f in food_domain
            if not any(kw in f.name.lower() for kw in PrimaryItemConstraint.ADDON_KEYWORDS)
            or any(base in f.name.lower() for base in ["bánh mì", "cơm", "mì", "bún", "phở"])
        ]
        stats["removed_by_addon"] = before_addon - len(food_domain)

        # Filter drink domain
        if not request.caffeine_allowed:
            before_caffeine = len(drink_domain)
            drink_domain = [d for d in drink_domain if d.category != Category.CA_PHE]
            stats["removed_by_caffeine"] = before_caffeine - len(drink_domain)

        stats["final_food_count"] = len(food_domain)
        stats["final_drink_count"] = len(drink_domain)

        return {
            CSPVariable.FOOD: food_domain,
            CSPVariable.DRINK: drink_domain
        }, stats

    def is_consistent(self, assignment: Assignment, request: UserRequest) -> bool:
        """
        Check if current assignment satisfies all hard constraints
        Used during backtracking search
        """
        for constraint in self.constraints:
            if constraint.is_hard and not constraint.is_satisfied(assignment, request):
                return False
        return True

    def get_hard_constraints_description(self) -> List[str]:
        """Get descriptions of all hard constraints for display"""
        return [f"{c.id}: {c.description}" for c in self.constraints if c.is_hard]
