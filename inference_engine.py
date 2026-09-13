"""
Inference Engine - Main AI System Integration
Implements f(X) - the complete AI recommendation function.

f(X) = argmax_{s ∈ C(X)} U(s, X)

where:
    X = user request (budget, meal time, preferences, etc.)
    C(X) = feasible solution set (all solutions satisfying hard constraints)
    U(s, X) = utility function evaluating soft preferences
    f(X) = optimal recommendation (highest utility feasible solution)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from csp import CSPModel, UserRequest
from backtracking_search import solve_csp, SearchResult
from utility import UtilityFunction, UtilityScore, rank_solutions
from knowledge_base import KnowledgeBase


@dataclass
class RecommendationResult:
    """
    Complete result of f(X) with full explanation trace.

    Contains:
    - Input X
    - CSP solving trace (domains, propagation, search)
    - Feasible set C(X)
    - Utility evaluations U(s, X) for each solution
    - Final ranked recommendations
    - The best solution f(X) = argmax U(s, X)
    """
    request: UserRequest
    search_result: SearchResult
    ranked_solutions: List[UtilityScore]
    best_solution: Optional[UtilityScore]

    def get_top_k(self, k: int = 5) -> List[UtilityScore]:
        """Get top-K recommendations"""
        return self.ranked_solutions[:k]

    def get_best(self) -> Optional[UtilityScore]:
        """Get f(X) - the optimal solution"""
        return self.best_solution


class InferenceEngine:
    """
    Main AI system implementing f(X).

    Pipeline:
    1. Input: X = user request
    2. Knowledge Base: retrieve menu items
    3. CSP Formulation: define variables, domains, constraints
    4. Constraint Propagation: reduce domains
    5. Backtracking Search: find C(X) = feasible solutions
    6. Utility Evaluation: compute U(s, X) for each s in C(X)
    7. Ranking: sort by utility score
    8. Output: f(X) = argmax U(s, X), and top-K recommendations
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.csp_model = CSPModel(knowledge_base)
        self.utility_function = UtilityFunction()

    def recommend(self, request: UserRequest, top_k: int = 5) -> RecommendationResult:
        """
        Main recommendation function: f(X)

        Implements the complete AI pipeline from user request to ranked recommendations.

        Args:
            request: User request X = (B, T, V, H, P, D)
            top_k: Number of top recommendations to return

        Returns:
            RecommendationResult containing f(X) and full explanation trace
        """

        # Step 1-4: CSP + Constraint Propagation + Backtracking Search
        # This produces C(X) = feasible solution set
        search_result = solve_csp(
            self.csp_model,
            request,
            max_solutions=None
        )

        # Step 5: Utility Evaluation and Ranking
        # For each s ∈ C(X), compute U(s, X) and rank
        ranked_solutions = rank_solutions(
            search_result.feasible_solutions,
            request,
            self.utility_function,
            top_k=top_k
        )

        # Step 6: Extract best solution f(X) = argmax_{s ∈ C(X)} U(s, X)
        best_solution = ranked_solutions[0] if ranked_solutions else None

        # Package complete result with full trace
        result = RecommendationResult(
            request=request,
            search_result=search_result,
            ranked_solutions=ranked_solutions,
            best_solution=best_solution
        )

        return result

    def set_utility_weights(self, weights: dict):
        """Update utility function weights (for advanced settings)"""
        self.utility_function.set_weights(weights)

    def get_utility_weights(self) -> dict:
        """Get current utility function weights"""
        return self.utility_function.weights.copy()


def format_complete_trace(result: RecommendationResult) -> str:
    """
    Format the complete inference trace showing the entire f(X) pipeline.

    This is used for the explanation/reasoning display in the UI.
    """
    trace = "# 🔬 CHI TIẾT QUÁ TRÌNH SUY LUẬN AI\n\n"

    # Step 1: Input X
    trace += "## BƯỚC 1: INPUT X\n\n"
    trace += "```\n"
    trace += f"X = {{\n"
    trace += f"  Budget (B): {result.request.budget:,} VND\n"
    trace += f"  MealTime (T): {result.request.meal_time.value}\n"
    trace += f"  Vegetarian (V): {result.request.vegetarian}\n"
    trace += f"  Hunger (H): {result.request.hunger_level}\n"
    trace += f"  Preferences (P): {result.request.taste_preferences}\n"
    trace += f"  WantsDrink (D): {result.request.wants_drink}\n"
    trace += f"  Thirsty: {result.request.is_thirsty}\n"
    trace += f"}}\n"
    trace += "```\n\n"

    trace += "---\n\n"
    trace += "## BƯỚC 2: CSP FORMULATION\n\n"
    trace += "**Variables:** Food, Drink\n\n"
    trace += f"**Initial Domains:**\n"
    trace += f"- D_Food: {result.search_result.propagation_stats.initial_food_domain} món\n"
    trace += f"- D_Drink: {result.search_result.propagation_stats.initial_drink_domain} đồ uống\n"
    trace += f"- Search space: {result.search_result.propagation_stats.initial_food_domain * (result.search_result.propagation_stats.initial_drink_domain if result.request.wants_drink else 1):,} tổ hợp\n\n"

    trace += "---\n\n"
    trace += "## BƯỚC 3: CONSTRAINT PROPAGATION\n\n"
    trace += "**Rút gọn miền giá trị bằng cách loại bỏ giá trị vi phạm ràng buộc:**\n\n"
    trace += f"- Sau lọc meal_time: {result.search_result.propagation_stats.after_time_filter} món ăn\n"
    trace += f"- Sau lọc diet: {result.search_result.propagation_stats.after_diet_filter} món ăn\n"
    trace += f"- Sau lọc budget: {result.search_result.propagation_stats.after_budget_filter} món ăn\n"
    trace += f"- Sau lọc add-on: {result.search_result.propagation_stats.after_addon_filter} món ăn\n"
    trace += f"- Sau lọc caffeine/drink: {result.search_result.propagation_stats.after_drink_filter} đồ uống\n\n"
    trace += f"- Sau lọc nhóm món: {result.search_result.propagation_stats.after_selection_filter} món ăn\n"
    trace += f"**Reduced search space:** {result.search_result.propagation_stats.after_selection_filter * (result.search_result.propagation_stats.after_drink_filter if result.request.wants_drink else 1):,} tổ hợp\n\n"

    trace += "---\n\n"
    trace += "## BƯỚC 4: BACKTRACKING SEARCH\n\n"
    trace += f"**Metrics:**\n"
    trace += f"- 🔍 Nodes visited: {result.search_result.search_metrics.nodes_visited}\n"
    trace += f"- ✅ Assignments tested: {result.search_result.search_metrics.assignments_tested}\n"
    trace += f"- ⚖️ Constraint checks: {result.search_result.search_metrics.constraint_checks}\n"
    trace += f"- ↩️ Backtracks: {result.search_result.search_metrics.backtracks}\n"
    trace += f"- ✔️ Valid solutions found: {result.search_result.search_metrics.solutions_found}\n\n"

    trace += f"**Tập nghiệm khả thi C(X):** {len(result.search_result.feasible_solutions)} nghiệm\n\n"

    trace += "---\n\n"
    trace += "## BƯỚC 5: UTILITY EVALUATION\n\n"
    trace += "**Đánh giá mỗi nghiệm s ∈ C(X) bằng hàm U(s, X)**\n\n"

    if result.ranked_solutions:
        trace += f"**Top 3 nghiệm có utility cao nhất:**\n\n"
        for idx, sol in enumerate(result.ranked_solutions[:3], 1):
            trace += f"{idx}. {sol.assignment.food.name}{' + ' + sol.assignment.drink.name if sol.assignment.drink else ''}\n"
            trace += f"   - Raw score: {sol.raw_score:.2f}\n"
            trace += f"   - Normalized: {sol.normalized_score:.1f}/100\n"
            trace += f"   - Price: {sol.assignment.get_total_price():,} VND\n\n"

    trace += "---\n\n"
    trace += "## BƯỚC 6: RANKING & OUTPUT\n\n"

    if result.ranked_solutions:
        best = result.ranked_solutions[0]
        trace += f"**f(X) = argmax U(s, X):**\n\n"
        trace += f"```\n"
        trace += f"f(X) = {best.assignment.food.name}{' + ' + best.assignment.drink.name if best.assignment.drink else ''}\n"
        trace += f"U(s, X) = {best.normalized_score:.1f}/100\n"
        trace += f"Price = {best.assignment.get_total_price():,} VND\n"
        trace += f"```\n\n"
        trace += f"**Top-K recommendations:** {len(result.ranked_solutions)} nghiệm được trả về\n"
    else:
        trace += "**Không tìm thấy nghiệm khả thi (C(X) = ∅)**\n"

    return trace
