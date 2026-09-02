"""
Heuristic Utility Evaluation Module
Implements U(s, X) - the utility function for ranking feasible solutions.
Evaluates soft preferences and computes scores for solution ranking.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from csp import Assignment, UserRequest
from knowledge_base import MenuItem, Category


@dataclass
class UtilityScore:
    """
    Detailed utility evaluation for a feasible solution
    U(s, X) = weighted sum of multiple criteria
    """
    assignment: Assignment
    total_score: float
    normalized_score: float  # 0-100 scale
    criteria_scores: Dict[str, float]
    criteria_reasons: Dict[str, str]

    # Add raw_score alias for backward compatibility with format_complete_trace
    @property
    def raw_score(self) -> float:
        """Alias for total_score"""
        return self.total_score

    def __lt__(self, other):
        """For sorting - higher scores are better"""
        return self.total_score > other.total_score


class UtilityFunction:
    """
    Implements U(s, X) - heuristic utility function for evaluating
    feasible solutions based on soft preferences.

    This is NOT a constraint checker - it assumes all solutions passed
    to it are already feasible (satisfy all hard constraints).

    The utility function ranks solutions by how well they match the
    user's soft preferences (hunger, taste, budget utilization, etc.)
    """

    def __init__(self):
        # Default weights for utility criteria
        self.weights = {
            "budget_fit": 0.30,        # How well price uses available budget
            "preference_match": 0.25,   # Match to taste preferences
            "variety_bonus": 0.20,      # Bonus for balanced food-drink pairing
            "hunger_fit": 0.15,         # Match to hunger level
            "thirst_fit": 0.10          # Match to thirst level
        }

    def set_weights(self, weights: Dict[str, float]):
        """Update utility function weights (for advanced settings)"""
        self.weights.update(weights)

    def evaluate(self, assignment: Assignment, request: UserRequest) -> UtilityScore:
        """
        Evaluate utility U(s, X) for a feasible solution s given request X

        Returns UtilityScore with detailed breakdown
        """
        criteria_scores = {}
        criteria_reasons = {}

        # Criterion 1: Budget fit
        budget_score, budget_reason = self._evaluate_budget_fit(
            assignment, request.budget
        )
        criteria_scores["budget_fit"] = budget_score
        criteria_reasons["budget_fit"] = budget_reason

        # Criterion 2: Preference match
        pref_score, pref_reason = self._evaluate_preference_match(
            assignment, request.taste_preferences
        )
        criteria_scores["preference_match"] = pref_score
        criteria_reasons["preference_match"] = pref_reason

        # Criterion 3: Variety/balance bonus
        variety_score, variety_reason = self._evaluate_variety_bonus(assignment)
        criteria_scores["variety_bonus"] = variety_score
        criteria_reasons["variety_bonus"] = variety_reason

        # Criterion 4: Hunger fit
        hunger_score, hunger_reason = self._evaluate_hunger_fit(
            assignment, request.hunger_level
        )
        criteria_scores["hunger_fit"] = hunger_score
        criteria_reasons["hunger_fit"] = hunger_reason

        # Criterion 5: Thirst fit
        thirst_score, thirst_reason = self._evaluate_thirst_fit(
            assignment, request.is_thirsty
        )
        criteria_scores["thirst_fit"] = thirst_score
        criteria_reasons["thirst_fit"] = thirst_reason

        # Compute weighted total: U(s, X) = Σ w_i * score_i
        total_score = sum(
            self.weights[criterion] * score
            for criterion, score in criteria_scores.items()
        )

        # Normalize to 0-100 scale for display
        normalized_score = total_score * 100

        return UtilityScore(
            assignment=assignment,
            total_score=total_score,
            normalized_score=normalized_score,
            criteria_scores=criteria_scores,
            criteria_reasons=criteria_reasons
        )

    def _evaluate_budget_fit(self, assignment: Assignment, budget: float) -> Tuple[float, str]:
        """
        Evaluate how well the solution uses available budget.
        Optimal: 70-90% utilization (not too cheap, not maxed out)
        """
        total_price = assignment.get_total_price()
        utilization = total_price / budget if budget > 0 else 0

        if 0.70 <= utilization <= 0.90:
            score = 1.0
            reason = f"Giá {int(total_price):,}đ tối ưu (~{int(utilization*100)}% ngân sách)"
        elif 0.50 <= utilization < 0.70:
            score = 0.7 + (utilization - 0.50) * 1.5
            reason = f"Giá {int(total_price):,}đ hợp lý ({int(utilization*100)}% ngân sách)"
        elif 0.90 < utilization <= 1.0:
            score = 0.9 - (utilization - 0.90) * 1.0
            reason = f"Giá {int(total_price):,}đ gần tối đa ({int(utilization*100)}% ngân sách)"
        else:  # < 0.50
            score = utilization * 1.2
            reason = f"Giá {int(total_price):,}đ tiết kiệm ({int(utilization*100)}% ngân sách)"

        return score, reason

    def _evaluate_preference_match(self,
                                  assignment: Assignment,
                                  taste_preferences: List[str]) -> Tuple[float, str]:
        """
        Evaluate match to user's taste/category preferences (soft preference)
        """
        if not taste_preferences:
            return 0.5, "Không có sở thích cụ thể"

        # Map preference strings to Category enum
        pref_map = {
            "cơm": Category.COM,
            "mì/bún/phở": Category.MI_BUN_PHO,
            "bánh mì": Category.BANH_MI,
            "nước ép": Category.NUOC_EP,
            "cà phê": Category.CA_PHE,
            "trà sữa": Category.TRA_SUA,
            "khác": Category.KHAC
        }

        preferred_categories = [
            pref_map[p] for p in taste_preferences if p in pref_map
        ]

        matches = []
        if assignment.food and assignment.food.category in preferred_categories:
            matches.append(assignment.food.category.value)
        if assignment.drink and assignment.drink.category in preferred_categories:
            matches.append(assignment.drink.category.value)

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

    def _evaluate_variety_bonus(self, assignment: Assignment) -> Tuple[float, str]:
        """
        Evaluate variety/balance in food-drink pairing.
        Good pairings get bonus (e.g., rice + juice, noodles + tea)
        """
        if not assignment.food or not assignment.drink:
            return 0.5, "Không có đồ uống"

        food_cat = assignment.food.category
        drink_cat = assignment.drink.category

        # Define good pairings based on common Vietnamese meal habits
        good_pairs = [
            (Category.COM, Category.NUOC_EP),
            (Category.COM, Category.TRA_SUA),
            (Category.MI_BUN_PHO, Category.TRA_SUA),
            (Category.MI_BUN_PHO, Category.NUOC_EP),
            (Category.BANH_MI, Category.CA_PHE),
            (Category.BANH_MI, Category.NUOC_EP)
        ]

        if (food_cat, drink_cat) in good_pairs:
            score = 1.0
            reason = f"Kết hợp cân bằng: {food_cat.value} + {drink_cat.value}"
        elif drink_cat == Category.KHAC:
            score = 0.6
            reason = "Đồ uống cơ bản (nước suối, nước ngọt...)"
        else:
            score = 0.5
            reason = f"Kết hợp: {food_cat.value} + {drink_cat.value}"

        return score, reason

    def _evaluate_hunger_fit(self,
                           assignment: Assignment,
                           hunger_level: str) -> Tuple[float, str]:
        """
        Evaluate how well food portion matches hunger level.
        Based on estimated calories (annotated in knowledge base).
        """
        if not assignment.food:
            return 0.0, "Không có món ăn"

        calories = assignment.food.calories_estimate

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

    def _evaluate_thirst_fit(self,
                           assignment: Assignment,
                           is_thirsty: bool) -> Tuple[float, str]:
        """
        Evaluate drink choice based on thirst level (soft preference)
        """
        if not assignment.drink:
            if is_thirsty:
                return 0.3, "Không có đồ uống khi khát"
            else:
                return 0.8, "Không cần đồ uống"

        drink = assignment.drink
        drink_price = drink.price

        if is_thirsty:
            # When thirsty, prefer refreshing drinks
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
            # When not thirsty, simple/cheap drinks are fine
            if drink_price <= 15000:
                score = 1.0
                reason = f"Đồ uống đơn giản ({int(drink_price):,}đ) khi không khát"
            else:
                score = 0.6
                reason = f"Đồ uống cao cấp khi không quá khát"

        return score, reason

    def format_score_breakdown(self, utility: UtilityScore) -> str:
        """Format detailed score breakdown for display"""
        lines = ["**Chi tiết điểm số U(s, X):**"]

        criteria_names = {
            "budget_fit": "Phù hợp giá",
            "preference_match": "Khớp sở thích",
            "variety_bonus": "Cân bằng dinh dưỡng",
            "hunger_fit": "Phù hợp mức đói",
            "thirst_fit": "Phù hợp mức khát"
        }

        for key, label in criteria_names.items():
            raw_score = utility.criteria_scores[key]
            weighted = raw_score * self.weights[key]
            lines.append(
                f"- **{label}**: {raw_score:.2f} × {self.weights[key]:.0%} "
                f"= {weighted:.3f}"
            )

        lines.append(f"\n**U(s, X) = {utility.total_score:.3f}**")
        lines.append(f"**Điểm chuẩn hóa: {utility.normalized_score:.1f}/100**")

        return "\n".join(lines)


def rank_solutions(feasible_solutions: List[Assignment],
                  request: UserRequest,
                  utility_function: UtilityFunction,
                  top_k: int = 5) -> List[UtilityScore]:
    """
    Rank feasible solutions using utility function U(s, X).

    For each s ∈ C(X):
        compute U(s, X)

    Return top-K solutions ranked by utility score.

    This implements the ranking component of f(X).
    """
    if not feasible_solutions:
        return []

    # Evaluate utility for each feasible solution
    scored_solutions = []
    for solution in feasible_solutions:
        utility = utility_function.evaluate(solution, request)
        scored_solutions.append(utility)

    # Sort by score (descending) and return top-K
    scored_solutions.sort(reverse=True, key=lambda x: x.total_score)

    return scored_solutions[:top_k]
