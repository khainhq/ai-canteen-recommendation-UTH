"""UI regression checks. Run with: python -m unittest test_ui -v"""
import unittest

from streamlit.testing.v1 import AppTest


class StudentInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file("app.py", default_timeout=30).run()

    def assert_no_errors(self):
        self.assertEqual(len(self.app.exception), 0)

    def submit_meal(self):
        next(b for b in self.app.button if "Gợi ý bữa ăn" in b.label).click().run()

    def test_initial_layout_and_guide(self):
        self.assert_no_errors()
        self.assertEqual([tab.label for tab in self.app.tabs],
                         ["🍽️ Chọn món", "📋 Menu căn tin", "🌷 Hướng dẫn"])
        previews = [m.value for m in self.app.markdown if 'class="menu-preview"' in m.value]
        self.assertEqual(len(previews), 2)
        self.assertTrue(all("data:image/jpeg;base64," in p for p in previews))
        self.assertTrue(any("BƯỚC 03" in m.value for m in self.app.markdown))

    def test_recommendations_persist_on_rerun(self):
        self.submit_meal()
        self.assert_no_errors()
        result = self.app.session_state["last_result"]
        self.assertTrue(result.ranked_solutions)
        self.assertTrue(all(s.assignment.get_total_price() <= 50000
                            for s in result.ranked_solutions))
        self.app.run()
        self.assert_no_errors()
        self.assertTrue(any('class="combo-card best"' in m.value for m in self.app.markdown))

    def test_lunch_preferences(self):
        self.app.slider[0].set_value(35000)
        self.app.radio[0].set_value("Trưa (sau 10:00)")
        self.app.multiselect[0].set_value(["Cơm"])
        self.submit_meal()
        self.assert_no_errors()
        result = self.app.session_state["last_result"]
        self.assertTrue(result.ranked_solutions)
        self.assertTrue(all(s.assignment.get_total_price() <= 35000
                            for s in result.ranked_solutions))

    def test_weight_validation_and_reset(self):
        for slider in self.app.slider:
            if slider.key and slider.key.startswith("weight_"):
                slider.set_value(0)
        next(b for b in self.app.button if b.label == "Lưu mức ưu tiên").click().run()
        self.assert_no_errors()
        self.assertTrue(self.app.error)
        next(b for b in self.app.button if b.label == "Khôi phục mặc định").click().run()
        self.assert_no_errors()
        self.assertEqual(self.app.slider(key="weight_budget_fit").value, 30)
        self.app.slider(key="weight_budget_fit").set_value(100)
        next(b for b in self.app.button if b.label == "Lưu mức ưu tiên").click().run()
        self.assert_no_errors()
        weights = self.app.session_state["inference_engine"].utility_function.weights
        self.assertAlmostEqual(sum(weights.values()), 1)


if __name__ == "__main__":
    unittest.main()
