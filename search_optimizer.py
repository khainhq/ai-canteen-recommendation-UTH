"""
Search and Multi-Criteria Optimization Module
Implements constrained combinatorial search over feasible combos with utility-based scoring.
Uses best-first search principles with explicit heuristic evaluation.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import heapq
from knowledge_base import MenuItem, Category


@dataclass
class ComboScore:
    """Represents a scored combo with detailed breakdown"""
    dish: MenuItem
    drink: MenuItem
    total_price: float
    total_score: float
    score_breakdown: Dict[str, float]
    reasoning: List[str]

    def __lt__(self, other):
        """For heap sorting - higher scores are better"""
        return self.total_score > other.total_score


class SearchOptimizer:
    """
    Implements state-space search with multi-criteria utility optimization.
    Each state = (dish, drink) combo; goal test = within budget + satisfies constraints.
    Heuristic = weighted utility function combining multiple criteria.
    """

    def __init__(self):
        # Default weights for utility function (can be adjusted by user)
        self.weights = {
            "price_fit": 0.30,      # How well price uses budget
            "preference_match": 0.25,  # Match to taste preferences
            "variety_bonus": 0.20,   # Bonus for diverse categories
            "hunger_fit": 0.15,      # Match to hunger level
            "thirst_fit": 0.10       # Match to thirst level
        }

    def set_weights(self, weights: Dict[str, float]):
        """Update utility function weights"""
        self.weights.update(weights)

    def _calculate_price_fit_score(self, combo_price: float, budget: float) -> Tuple[float, str]:
        """
        Score how well the combo uses available budget.
        Optimal: use 70-90% of budget (not too cheap, not maxed out)
        """
        utilization = combo_price / budget if budget > 0 else 0

        if 0.70 <= utilization <= 0.90:
            score = 1.0
            reason = f"Giá {int(combo_price):,}đ tối ưu (~{int(utilization*100)}% ngân sách)"
        elif 0.50 <= utilization < 0.70:
            score = 0.7 + (utilization - 0.50) * 1.5
            reason = f"Giá {int(combo_price):,}đ hợp lý ({int(utilization*100)}% ngân sách)"
        elif 0.90 < utilization <= 1.0:
            score = 0.9 - (utilization - 0.90) * 1.0
            reason = f"Giá {int(combo_price):,}đ gần tối đa ({int(utilization*100)}% ngân sách)"
        else:  # < 0.50
            score = utilization * 1.2
            reason = f"Giá {int(combo_price):,}đ tiết kiệm ({int(utilization*100)}% ngân sách)"

        return score, reason

    def _calculate_preference_match_score(self,
                                         dish: MenuItem,
                                         drink: MenuItem,
                                         taste_preferences: List[str],
                                         taste_boost: float) -> Tuple[float, str]:
        """
        Score based on matching user's taste/category preferences.
        """
        if not taste_preferences or taste_boost == 0:
            return 0.5, "Không có sở thích cụ thể"

        # Convert string preferences to Category enum
        pref_map = {
            "cơm": Category.COM,
            "mì/bún/phở": Category.MI_BUN_PHO,
            "bánh mì": Category.BANH_MI,
            "nước ép": Category.NUOC_EP,
            "cà phê": Category.CA_PHE,
            "trà sữa": Category.TRA_SUA,
            "khác": Category.KHAC
        }

        preferred_categories = [pref_map.get(p) for p in taste_preferences if p in pref_map]

        matches = []
        if dish.category in preferred_categories:
            matches.append(dish.category.value)
        if drink.category in preferred_categories:
            matches.append(drink.category.value)

        if len(matches) == 2:
            score = 1.0
            reason = f"Cả món ăn và đồ uống đều khớp sở thích: {', '.join(matches)}"
        elif len(matches) == 1:
            score = 0.7
            reason = f"Khớp sở thích: {matches[0]}"
        else:
            score = 0.3
            reason = "Không khớp sở thích đã chọn"

        return score, reason

    def _calculate_variety_bonus(self,
                                dish: MenuItem,
                                drink: MenuItem) -> Tuple[float, str]:
        """
        Bonus for diverse food-drink pairing (e.g., rice + juice is more balanced than rice + coffee)
        """
        dish_cat = dish.category
        drink_cat = drink.category

        # Good pairings
        good_pairs = [
            (Category.COM, Category.NUOC_EP),
            (Category.COM, Category.TRA_SUA),
            (Category.MI_BUN_PHO, Category.TRA_SUA),
            (Category.MI_BUN_PHO, Category.NUOC_EP),
            (Category.BANH_MI, Category.CA_PHE),
            (Category.BANH_MI, Category.NUOC_EP)
        ]

        if (dish_cat, drink_cat) in good_pairs:
            score = 1.0
            reason = f"Kết hợp cân bằng: {dish_cat.value} + {drink_cat.value}"
        elif drink_cat == Category.KHAC:
            score = 0.6
            reason = "Đồ uống cơ bản (nước suối, nước ngọt...)"
        else:
            score = 0.5
            reason = f"Kết hợp: {dish_cat.value} + {drink_cat.value}"

        return score, reason

    def _calculate_hunger_fit_score(self,
                                   dish: MenuItem,
                                   hunger_level: str,
                                   hunger_weights: Dict[str, float]) -> Tuple[float, str]:
        """
        Score how well the dish matches hunger level based on estimated calories.
        """
        calories = dish.calories_estimate

        if hunger_level == "light":
            if calories < 400:
                score = 1.0
                reason = f"Món nhẹ ({calories} kcal) phù hợp khi đói ít"
            elif calories < 500:
                score = 0.7
                reason = f"Món vừa phải ({calories} kcal)"
            else:
                score = 0.4
                reason = f"Món hơi no ({calories} kcal) cho mức đói nhẹ"

        elif hunger_level == "very_hungry":
            if calories >= 500:
                score = 1.0
                reason = f"Món no ({calories} kcal) phù hợp khi rất đói"
            elif calories >= 400:
                score = 0.7
                reason = f"Món vừa phải ({calories} kcal)"
            else:
                score = 0.4
                reason = f"Món nhẹ ({calories} kcal) cho mức rất đói"

        else:  # normal
            if 400 <= calories <= 500:
                score = 1.0
                reason = f"Món vừa đủ ({calories} kcal) cho mức đói bình thường"
            else:
                score = 0.7
                reason = f"Món ăn {calories} kcal"

        return score, reason

    def _calculate_thirst_fit_score(self,
                                   drink: MenuItem,
                                   is_thirsty: bool,
                                   cheap_drink_boost: float) -> Tuple[float, str]:
        """
        Score drink choice based on thirst level.
        """
        drink_price = drink.price

        if is_thirsty:
            # When thirsty, prefer more substantial/refreshing drinks
            if drink.category in [Category.NUOC_EP, Category.TRA_SUA]:
                score = 1.0
                reason = f"Đồ uống giải khát tốt: {drink.name}"
            elif drink_price <= 15000:
                score = 0.5
                reason = "Đồ uống cơ bản khi khát nước"
            else:
                score = 0.7
                reason = f"Đồ uống: {drink.name}"
        else:
            # When not thirsty, cheap/simple drinks are fine
            if drink_price <= 15000:
                score = 1.0
                reason = f"Đồ uống đơn giản ({int(drink_price):,}đ) khi không khát"
            else:
                score = 0.6
                reason = f"Đồ uống cao cấp khi không quá khát"

        return score, reason

    def search_and_score(self,
                        feasible_combos: List[Tuple[MenuItem, MenuItem]],
                        budget: float,
                        is_thirsty: bool,
                        hunger_level: str,
                        taste_preferences: List[str],
                        hunger_weights: Dict[str, float],
                        cheap_drink_boost: float,
                        taste_boost: float,
                        top_k: int = 5) -> List[ComboScore]:
        """
        Main search algorithm: evaluate all feasible combos using multi-criteria utility function.
        Implements best-first search by scoring all states and returning top-K.
        """
        if not feasible_combos:
            return []

        scored_combos = []

        for dish, drink in feasible_combos:
            combo_price = dish.price + drink.price

            # Calculate individual criterion scores
            price_score, price_reason = self._calculate_price_fit_score(combo_price, budget)
            pref_score, pref_reason = self._calculate_preference_match_score(
                dish, drink, taste_preferences, taste_boost
            )
            variety_score, variety_reason = self._calculate_variety_bonus(dish, drink)
            hunger_score, hunger_reason = self._calculate_hunger_fit_score(
                dish, hunger_level, hunger_weights
            )
            thirst_score, thirst_reason = self._calculate_thirst_fit_score(
                drink, is_thirsty, cheap_drink_boost
            )

            # Weighted utility function
            total_score = (
                self.weights["price_fit"] * price_score +
                self.weights["preference_match"] * pref_score +
                self.weights["variety_bonus"] * variety_score +
                self.weights["hunger_fit"] * hunger_score +
                self.weights["thirst_fit"] * thirst_score
            )

            # Normalize to 0-100 scale for display
            total_score_normalized = total_score * 100

            score_breakdown = {
                "price_fit": price_score,
                "preference_match": pref_score,
                "variety_bonus": variety_score,
                "hunger_fit": hunger_score,
                "thirst_fit": thirst_score
            }

            reasoning = [
                price_reason,
                pref_reason,
                variety_reason,
                hunger_reason,
                thirst_reason
            ]

            scored_combo = ComboScore(
                dish=dish,
                drink=drink,
                total_price=combo_price,
                total_score=total_score_normalized,
                score_breakdown=score_breakdown,
                reasoning=reasoning
            )

            scored_combos.append(scored_combo)

        # Use heap to efficiently get top-K (best-first search)
        top_combos = heapq.nsmallest(top_k, scored_combos)

        return top_combos

    def format_score_breakdown(self, combo: ComboScore) -> str:
        """Format score breakdown for display in Vietnamese"""
        lines = ["**Chi tiết điểm số:**"]

        criteria_names = {
            "price_fit": "Phù hợp giá",
            "preference_match": "Khớp sở thích",
            "variety_bonus": "Cân bằng dinh dưỡng",
            "hunger_fit": "Phù hợp mức đói",
            "thirst_fit": "Phù hợp mức khát"
        }

        for key, label in criteria_names.items():
            raw_score = combo.score_breakdown[key]
            weighted = raw_score * self.weights[key] * 100
            lines.append(
                f"- **{label}**: {raw_score:.2f} × {self.weights[key]:.0%} "
                f"= {weighted:.1f} điểm"
            )

        lines.append(f"\n**Tổng điểm**: {combo.total_score:.1f}/100")

        return "\n".join(lines)
