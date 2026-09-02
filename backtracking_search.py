"""
Backtracking Search Module
Implements recursive backtracking algorithm for CSP solution finding.
Searches the assignment space systematically with constraint checking and backtracking.
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from csp import CSPModel, Assignment, UserRequest, CSPVariable
from knowledge_base import MenuItem


@dataclass
class SearchMetrics:
    """
    Tracks search algorithm statistics for explainability
    """
    nodes_visited: int = 0
    assignments_tested: int = 0
    constraint_checks: int = 0
    backtracks: int = 0
    solutions_found: int = 0

    def reset(self):
        """Reset all metrics for a new search"""
        self.nodes_visited = 0
        self.assignments_tested = 0
        self.constraint_checks = 0
        self.backtracks = 0
        self.solutions_found = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            "Nodes visited": self.nodes_visited,
            "Assignments tested": self.assignments_tested,
            "Constraint checks": self.constraint_checks,
            "Backtracks": self.backtracks,
            "Solutions found": self.solutions_found
        }


class BacktrackingSearch:
    """
    Implements backtracking search algorithm for finding feasible CSP solutions.

    The algorithm:
    1. Select an unassigned variable
    2. Try each value from its domain
    3. If value is consistent with current assignment:
       - Assign the value
       - Recursively search from this state
       - If search fails, backtrack (undo assignment)
    4. Return solutions

    This is the core AI search technique demonstrating:
    - State-space exploration
    - Constraint satisfaction
    - Recursive backtracking
    - Pruning invalid branches
    """

    def __init__(self, csp_model: CSPModel):
        self.csp = csp_model
        self.metrics = SearchMetrics()
        self.max_solutions = 100  # Limit solutions to avoid excessive computation
        self.solutions: List[Assignment] = []

    def search(self,
              domains: Dict[CSPVariable, List[MenuItem]],
              request: UserRequest,
              max_solutions: int = 100) -> Tuple[List[Assignment], SearchMetrics]:
        """
        Main entry point for backtracking search.

        Returns: (list of feasible solutions, search metrics)
        """
        self.solutions = []
        self.metrics.reset()
        self.max_solutions = max_solutions

        # Start backtracking with empty assignment
        initial_assignment = Assignment()
        self._backtrack(initial_assignment, domains, request)

        return self.solutions, self.metrics

    def _backtrack(self,
                  assignment: Assignment,
                  domains: Dict[CSPVariable, List[MenuItem]],
                  request: UserRequest) -> bool:
        """
        Recursive backtracking algorithm.

        Returns True if at least one solution found (for optimization),
        False to continue searching for more solutions.
        """
        self.metrics.nodes_visited += 1

        # Base case: assignment is complete
        if assignment.is_complete(request.wants_drink):
            # Found a feasible solution
            self.solutions.append(assignment.copy())
            self.metrics.solutions_found += 1

            # Continue searching for more solutions (don't return True)
            if len(self.solutions) >= self.max_solutions:
                return True  # Stop when max solutions reached
            return False

        # Recursive case: select unassigned variable
        var = self._select_unassigned_variable(assignment, request)

        if var is None:
            # No variable to assign (shouldn't happen if is_complete works correctly)
            return False

        # Try each value in the domain
        for value in domains[var]:
            self.metrics.assignments_tested += 1

            # Create new assignment with this value
            new_assignment = assignment.copy()

            if var == CSPVariable.FOOD:
                new_assignment.food = value
            elif var == CSPVariable.DRINK:
                new_assignment.drink = value

            # Check if assignment is consistent with constraints
            self.metrics.constraint_checks += 1
            if self.csp.is_consistent(new_assignment, request):
                # Assignment is valid, recurse
                result = self._backtrack(new_assignment, domains, request)

                if result:  # Max solutions reached
                    return True
            else:
                # Constraint violated, backtrack
                self.metrics.backtracks += 1

        # All values tried for this variable, backtrack
        return False

    def _select_unassigned_variable(self,
                                   assignment: Assignment,
                                   request: UserRequest) -> Optional[CSPVariable]:
        """
        Select next variable to assign.

        Strategy: assign Food first, then Drink
        (This is a simple static ordering; more sophisticated heuristics
        like MRV - Minimum Remaining Values - could be used but would
        complicate the explanation during defense)
        """
        if assignment.food is None:
            return CSPVariable.FOOD

        if request.wants_drink and assignment.drink is None:
            return CSPVariable.DRINK

        return None


class SearchResult:
    """
    Complete result of CSP search + constraint propagation
    Contains both the process trace and final results
    """

    def __init__(self,
                 request: UserRequest,
                 initial_domains: Dict[CSPVariable, List[MenuItem]],
                 reduced_domains: Dict[CSPVariable, List[MenuItem]],
                 propagation_stats: Dict,
                 feasible_solutions: List[Assignment],
                 search_metrics: SearchMetrics):
        self.request = request
        self.initial_domains = initial_domains
        self.reduced_domains = reduced_domains
        self.search_metrics = search_metrics
        self.feasible_solutions = feasible_solutions

        # Create propagation_stats object with attributes for easy access
        class PropagationStats:
            def __init__(self, stats_dict):
                self.initial_food_domain = stats_dict['initial_food_count']
                self.initial_drink_domain = stats_dict['initial_drink_count']
                self.after_time_filter = stats_dict['final_food_count']  # approximation
                self.after_diet_filter = stats_dict['final_food_count']
                self.after_budget_filter = stats_dict['final_food_count']
                self.after_addon_filter = stats_dict['final_food_count']
                self.after_drink_filter = stats_dict['final_drink_count']

        self.propagation_stats = PropagationStats(propagation_stats)
        self._raw_propagation_stats = propagation_stats

    def get_feasible_set_size(self) -> int:
        """Get |C(X)| - size of feasible solution set"""
        return len(self.feasible_solutions)

    def format_propagation_summary(self) -> str:
        """Format constraint propagation summary for display"""
        stats = self._raw_propagation_stats
        lines = [
            "**Bước 1: Miền ban đầu (Initial Domains)**",
            f"- Món ăn: {stats['initial_food_count']} items",
            f"- Đồ uống: {stats['initial_drink_count']} items",
            "",
            "**Bước 2: Lan truyền ràng buộc (Constraint Propagation)**"
        ]

        if stats['removed_by_meal_time'] > 0:
            lines.append(f"- Loại bỏ {stats['removed_by_meal_time']} món không đúng giờ")

        if stats['removed_by_diet'] > 0:
            lines.append(f"- Loại bỏ {stats['removed_by_diet']} món không phù hợp chế độ ăn")

        if stats['removed_by_budget'] > 0:
            lines.append(f"- Loại bỏ {stats['removed_by_budget']} món quá ngân sách")

        if stats['removed_by_caffeine'] > 0:
            lines.append(f"- Loại bỏ {stats['removed_by_caffeine']} đồ uống có caffeine")

        if stats['removed_by_addon'] > 0:
            lines.append(f"- Loại bỏ {stats['removed_by_addon']} addon không hợp lệ")

        lines.extend([
            "",
            "**Bước 3: Miền sau khi rút gọn (Reduced Domains)**",
            f"- Món ăn: {stats['final_food_count']} items",
            f"- Đồ uống: {stats['final_drink_count']} items"
        ])

        return "\n".join(lines)

        lines.extend([
            "",
            "**Bước 3: Miền sau khi rút gọn (Reduced Domains)**",
            f"- Món ăn: {stats['final_food_count']} items",
            f"- Đồ uống: {stats['final_drink_count']} items"
        ])

        return "\n".join(lines)

    def format_search_summary(self) -> str:
        """Format backtracking search summary for display"""
        metrics = self.search_metrics
        lines = [
            "**Bước 4: Tìm kiếm Backtracking (Backtracking Search)**",
            f"- Nodes đã duyệt: {metrics.nodes_visited}",
            f"- Phép gán thử: {metrics.assignments_tested}",
            f"- Kiểm tra ràng buộc: {metrics.constraint_checks}",
            f"- Số lần quay lui: {metrics.backtracks}",
            f"- Nghiệm tìm được: {metrics.solutions_found}",
            "",
            f"**|C(X)| = {self.get_feasible_set_size()}** nghiệm khả thi"
        ]

        return "\n".join(lines)


def solve_csp(csp_model: CSPModel,
             request: UserRequest,
             max_solutions: int = 100) -> SearchResult:
    """
    Complete CSP solving pipeline:
    1. Get initial domains
    2. Apply constraint propagation
    3. Run backtracking search
    4. Return SearchResult with complete trace

    This function implements the constraint satisfaction and search
    components of f(X).
    """
    # Step 1: Get initial domains
    initial_domains = csp_model.get_initial_domains(request)

    # Step 2: Constraint propagation (domain reduction)
    reduced_domains, propagation_stats = csp_model.propagate_constraints(
        initial_domains,
        request
    )

    # Step 3: Backtracking search
    search = BacktrackingSearch(csp_model)
    feasible_solutions, search_metrics = search.search(
        reduced_domains,
        request,
        max_solutions
    )

    # Step 4: Package results
    result = SearchResult(
        request=request,
        initial_domains=initial_domains,
        reduced_domains=reduced_domains,
        propagation_stats=propagation_stats,
        feasible_solutions=feasible_solutions,
        search_metrics=search_metrics
    )

    return result
