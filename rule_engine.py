"""
Rule-Based Filtering Engine
Implements production rules (IF-THEN) using forward chaining to filter infeasible items/combos.
Each rule is explicitly named and traceable for the reasoning explanation.
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from knowledge_base import MenuItem, MealTime, DietType, Category


@dataclass
class Rule:
    """Production rule representation: IF condition THEN action"""
    id: str
    description: str
    priority: int  # Lower number = higher priority


@dataclass
class RuleFiringRecord:
    """Record of a rule that was fired during inference"""
    rule_id: str
    rule_description: str
    items_filtered: int


class RuleEngine:
    """
    Forward chaining rule-based system for filtering infeasible menu items and combos.
    Rules are applied in priority order to progressively narrow the search space.
    """

    def __init__(self):
        self.rules = self._define_rules()
        self.firing_trace: List[RuleFiringRecord] = []

    def _define_rules(self) -> List[Rule]:
        """Define all production rules explicitly"""
        return [
            Rule("R1", "IF meal_time = breakfast THEN exclude all lunch-only items", 1),
            Rule("R2", "IF meal_time = lunch THEN exclude all breakfast-only items", 1),
            Rule("R3", "IF diet = chay THEN exclude all mặn-only items", 2),
            Rule("R4", "IF diet = mặn THEN exclude all chay-only items", 2),
            Rule("R5", "IF dish_price + drink_price > budget THEN exclude combo", 3),
            Rule("R6", "IF thirsty = no THEN prioritize cheap drinks (water, etc.)", 4),
            Rule("R7", "IF hunger = light THEN prefer lower-calorie dishes", 5),
            Rule("R8", "IF hunger = very_hungry THEN prefer higher-calorie dishes", 5),
            Rule("R9", "IF taste_preferences specified THEN boost matching categories", 6),
            Rule("R10", "IF no drink needed AND thirsty = no THEN allow no-drink combo", 7)
        ]

    def reset_trace(self):
        """Clear the firing trace for a new inference session"""
        self.firing_trace = []

    def apply_meal_time_rules(self, dishes: List[MenuItem],
                              meal_time: MealTime) -> List[MenuItem]:
        """
        Apply R1 and R2: filter dishes by meal time availability
        """
        initial_count = len(dishes)

        if meal_time == MealTime.BREAKFAST:
            filtered = [d for d in dishes if d.meal_time == MealTime.BREAKFAST]
            self.firing_trace.append(RuleFiringRecord(
                "R1",
                "IF meal_time = breakfast THEN exclude all lunch-only items",
                initial_count - len(filtered)
            ))
        else:  # LUNCH
            filtered = [d for d in dishes if d.meal_time == MealTime.LUNCH]
            self.firing_trace.append(RuleFiringRecord(
                "R2",
                "IF meal_time = lunch THEN exclude all breakfast-only items",
                initial_count - len(filtered)
            ))

        return filtered

    def apply_diet_rules(self, items: List[MenuItem],
                         diet: DietType) -> List[MenuItem]:
        """
        Apply R3 and R4: filter items by dietary preference
        """
        initial_count = len(items)
        filtered = [item for item in items if item.matches_diet(diet)]

        if diet == DietType.CHAY:
            filtered_count = initial_count - len(filtered)
            if filtered_count > 0:
                self.firing_trace.append(RuleFiringRecord(
                    "R3",
                    "IF diet = chay THEN exclude all mặn-only items",
                    filtered_count
                ))
        elif diet == DietType.MAN:
            filtered_count = initial_count - len(filtered)
            if filtered_count > 0:
                self.firing_trace.append(RuleFiringRecord(
                    "R4",
                    "IF diet = mặn THEN exclude all chay-only items",
                    filtered_count
                ))

        return filtered

    def apply_budget_rule(self, combos: List[Tuple[MenuItem, MenuItem]],
                         budget: float) -> List[Tuple[MenuItem, MenuItem]]:
        """
        Apply R5: filter combos that exceed budget
        """
        initial_count = len(combos)
        filtered = [
            (dish, drink) for dish, drink in combos
            if dish.price + drink.price <= budget
        ]

        filtered_count = initial_count - len(filtered)
        if filtered_count > 0:
            self.firing_trace.append(RuleFiringRecord(
                "R5",
                f"IF dish_price + drink_price > {int(budget)} VND THEN exclude combo",
                filtered_count
            ))

        return filtered

    def apply_thirst_rule(self, drinks: List[MenuItem],
                         is_thirsty: bool) -> Tuple[List[MenuItem], float]:
        """
        Apply R6: if not thirsty, prioritize cheap/simple drinks
        Returns (filtered_drinks, priority_boost_for_cheap)
        """
        if not is_thirsty:
            self.firing_trace.append(RuleFiringRecord(
                "R6",
                "IF thirsty = no THEN prioritize cheap drinks (water, basic drinks)",
                0  # No items filtered, but affects scoring
            ))
            return drinks, 0.3  # Boost cheap drinks in scoring

        return drinks, 0.0

    def apply_hunger_rules(self, dishes: List[MenuItem],
                          hunger_level: str) -> Tuple[List[MenuItem], Dict[str, float]]:
        """
        Apply R7 and R8: adjust preferences based on hunger level
        Returns (dishes, calorie_preference_weights)
        """
        weights = {"light": 0.0, "normal": 0.0, "very_hungry": 0.0}

        if hunger_level == "light":
            weights["light"] = 0.2
            self.firing_trace.append(RuleFiringRecord(
                "R7",
                "IF hunger = light THEN prefer lower-calorie dishes",
                0
            ))
        elif hunger_level == "very_hungry":
            weights["very_hungry"] = 0.2
            self.firing_trace.append(RuleFiringRecord(
                "R8",
                "IF hunger = very_hungry THEN prefer higher-calorie dishes",
                0
            ))

        return dishes, weights

    def apply_taste_preference_rule(self, taste_preferences: List[str]) -> float:
        """
        Apply R9: boost score for items matching taste preferences
        Returns the boost weight to use in scoring
        """
        if taste_preferences and len(taste_preferences) > 0:
            self.firing_trace.append(RuleFiringRecord(
                "R9",
                f"IF taste_preferences = {taste_preferences} THEN boost matching categories",
                0
            ))
            return 0.25

        return 0.0

    def filter_items(self,
                    dishes: List[MenuItem],
                    drinks: List[MenuItem],
                    meal_time: MealTime,
                    diet: DietType,
                    budget: float,
                    is_thirsty: bool,
                    hunger_level: str,
                    taste_preferences: List[str]) -> Dict:
        """
        Main entry point: apply all filtering rules in forward chaining sequence.
        Returns filtered items and metadata for scoring.
        """
        self.reset_trace()

        # Phase 1: Filter dishes by meal time (R1/R2)
        filtered_dishes = self.apply_meal_time_rules(dishes, meal_time)

        # Phase 2: Filter by dietary preference (R3/R4)
        filtered_dishes = self.apply_diet_rules(filtered_dishes, diet)

        # Phase 3: Apply hunger-based preferences (R7/R8)
        filtered_dishes, hunger_weights = self.apply_hunger_rules(
            filtered_dishes, hunger_level
        )

        # Phase 4: Apply thirst-based drink preferences (R6)
        filtered_drinks, cheap_drink_boost = self.apply_thirst_rule(
            drinks, is_thirsty
        )

        # Phase 5: Apply taste preference boost (R9)
        taste_boost = self.apply_taste_preference_rule(taste_preferences)

        # Phase 6: Generate all feasible combos
        all_combos = [
            (dish, drink)
            for dish in filtered_dishes
            for drink in filtered_drinks
        ]

        # Phase 7: Filter combos by budget (R5)
        feasible_combos = self.apply_budget_rule(all_combos, budget)

        return {
            "combos": feasible_combos,
            "dishes": filtered_dishes,
            "drinks": filtered_drinks,
            "hunger_weights": hunger_weights,
            "cheap_drink_boost": cheap_drink_boost,
            "taste_boost": taste_boost,
            "rules_fired": self.firing_trace.copy()
        }

    def get_firing_trace(self) -> List[RuleFiringRecord]:
        """Get the complete trace of rules fired during last inference"""
        return self.firing_trace.copy()

    def format_trace_for_display(self) -> str:
        """Format firing trace as human-readable Vietnamese text"""
        if not self.firing_trace:
            return "Không có luật nào được kích hoạt."

        lines = ["**Các luật đã được áp dụng:**"]
        for record in self.firing_trace:
            if record.items_filtered > 0:
                lines.append(
                    f"- ✓ **{record.rule_id}**: {record.rule_description} "
                    f"→ Loại bỏ {record.items_filtered} món"
                )
            else:
                lines.append(
                    f"- ✓ **{record.rule_id}**: {record.rule_description}"
                )

        return "\n".join(lines)
