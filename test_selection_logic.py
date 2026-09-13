"""Regression tests for strict selections and complete CSP enumeration."""
import itertools
import unittest

from csp import Assignment, CSPModel, UserRequest
from inference_engine import InferenceEngine, format_complete_trace
from knowledge_base import Category, KnowledgeBase, MealTime


class SelectionLogicTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()
        self.engine = InferenceEngine(self.kb)

    def test_classification(self):
        for food in self.kb.dishes:
            if food.name.startswith('Bánh mì'):
                self.assertEqual(food.category, Category.BANH_MI)
        self.assertEqual(self.kb._categorize_by_name('Hảo hảo ốp la'), Category.MI_BUN_PHO)
        for drink in self.kb.drinks:
            if drink.category == Category.TRA_SUA:
                self.assertIn('trà sữa', drink.name.lower())

    def test_request_matrix_against_independent_enumeration(self):
        for budget, time, vegetarian, water, prefs in itertools.product(
            [20000, 50000, 100000], list(MealTime), [False, True], [False, True],
            [[], ['mì/bún/phở'], ['bánh mì'], ['cơm'], ['nước ép'],
             ['mì/bún/phở', 'bánh mì', 'trà sữa']]
        ):
            with self.subTest(budget=budget, time=time, vegetarian=vegetarian, water=water, prefs=prefs):
                request = UserRequest(budget, time, vegetarian, 'normal', prefs,
                                      water, is_thirsty=water)
                expected = set()
                foods = {p for p in prefs if p in ['cơm', 'mì/bún/phở', 'bánh mì']}
                drinks = set(prefs) - foods
                for food in self.kb.dishes:
                    if food.meal_time != time:
                        continue
                    if vegetarian and food.diet.value not in ['chay', 'cả hai']:
                        continue
                    if foods and food.category.value not in foods:
                        continue
                    for drink in (self.kb.drinks if water else [None]):
                        if drink and drinks and drink.category.value not in drinks:
                            continue
                        if food.price + (drink.price if drink else 0) <= budget:
                            expected.add((food.name, drink.name if drink else None))
                result = self.engine.recommend(request)
                actual = {(s.food.name, s.drink.name if s.drink else None)
                          for s in result.search_result.feasible_solutions}
                self.assertEqual(actual, expected)
                self.assertEqual(len(result.ranked_solutions), min(5, len(expected)))
                self.assertTrue(all(self.engine.csp_model.is_consistent(s, request)
                                    for s in result.search_result.feasible_solutions))
                format_complete_trace(result)  # Must handle food-only and empty results.

    def test_no_early_cutoff(self):
        request = UserRequest(100000, MealTime.BREAKFAST, False, 'normal', [], True)
        result = self.engine.recommend(request)
        self.assertGreater(len(result.search_result.feasible_solutions), 100)
        scores = [self.engine.utility_function.evaluate(s, request).total_score
                  for s in result.search_result.feasible_solutions]
        self.assertEqual(result.best_solution.total_score, max(scores))

    def test_constraint_rejects_unwanted_drink(self):
        request = UserRequest(100000, MealTime.BREAKFAST, False, 'normal', [], False)
        self.assertFalse(CSPModel(self.kb).is_consistent(
            Assignment(self.kb.dishes[0], self.kb.drinks[0]), request))


if __name__ == '__main__':
    unittest.main()
