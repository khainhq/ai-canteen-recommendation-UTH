#!/usr/bin/env python3
"""
Pre-Defense Verification Script
Run this before your defense to ensure everything works.
"""

import sys
from pathlib import Path

def verify_files():
    """Check all required files exist"""
    print("="*60)
    print("STEP 1: Verifying Project Files")
    print("="*60)

    required_files = {
        "Core AI Modules": [
            "knowledge_base.py",
            "csp.py",
            "backtracking_search.py",
            "utility.py",
            "inference_engine.py"
        ],
        "Application": [
            "app.py"
        ],
        "Documentation": [
            "README.md",
            "REFACTORING_SUMMARY.md",
            "QUICK_START.md",
            "ACCEPTANCE_CRITERIA.md",
            "PROJECT_COMPLETE.md"
        ],
        "Testing": [
            "test_system.py"
        ]
    }

    all_ok = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = Path(file).exists()
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {file}")
            if not exists:
                all_ok = False

    return all_ok


def verify_imports():
    """Check all modules can be imported"""
    print("\n" + "="*60)
    print("STEP 2: Verifying Module Imports")
    print("="*60)

    modules = [
        ("knowledge_base", "KnowledgeBase"),
        ("csp", "CSPModel, UserRequest, MealTime"),
        ("backtracking_search", "solve_csp, BacktrackingSearch"),
        ("utility", "UtilityFunction, rank_solutions"),
        ("inference_engine", "InferenceEngine, format_complete_trace")
    ]

    all_ok = True
    for module, items in modules:
        try:
            exec(f"from {module} import {items}")
            print(f"  [OK] {module}")
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            all_ok = False

    return all_ok


def verify_system():
    """Test the complete system"""
    print("\n" + "="*60)
    print("STEP 3: Testing System Functionality")
    print("="*60)

    try:
        from knowledge_base import KnowledgeBase
        from csp import UserRequest, MealTime, CSPModel
        from inference_engine import InferenceEngine

        # Initialize
        kb = KnowledgeBase()
        engine = InferenceEngine(kb)

        print(f"  [OK] System initialized")
        print(f"       - Dishes: {len(kb.get_all_dishes())}")
        print(f"       - Drinks: {len(kb.get_all_drinks())}")

        # Test recommendation
        request = UserRequest(
            budget=40000,
            meal_time=MealTime.BREAKFAST,
            vegetarian=False,
            hunger_level='normal',
            taste_preferences=['bánh mì'],
            wants_drink=True,
            is_thirsty=True,
            caffeine_allowed=True
        )

        result = engine.recommend(request, top_k=5)

        print(f"  [OK] f(X) pipeline executed")
        print(f"       - Nodes visited: {result.search_result.search_metrics.nodes_visited}")
        print(f"       - Backtracks: {result.search_result.search_metrics.backtracks}")
        print(f"       - |C(X)| = {len(result.search_result.feasible_solutions)}")

        if result.best_solution:
            print(f"  [OK] Best solution found")
            print(f"       - Utility score: {result.best_solution.normalized_score:.1f}/100")
            print(f"       - Price: {result.best_solution.assignment.get_total_price():,.0f} VND")
        else:
            print(f"  [WARN] No feasible solution (C(X) = empty)")

        # Verify backtracking occurred
        if result.search_result.search_metrics.backtracks > 0:
            print(f"  [OK] Backtracking verified (not fake)")
        else:
            print(f"  [WARN] No backtracks in this query")

        return True

    except Exception as e:
        print(f"  [FAIL] System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_csp_formulation():
    """Check CSP is properly formulated"""
    print("\n" + "="*60)
    print("STEP 4: Verifying CSP Formulation")
    print("="*60)

    try:
        from csp import CSPModel, CSPVariable
        from knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        csp = CSPModel(kb)

        # Check variables
        print(f"  [OK] CSP variables: {[v.value for v in CSPVariable]}")

        # Check constraints
        constraints = csp.get_hard_constraints_description()
        print(f"  [OK] Hard constraints: {len(constraints)}")
        for c in constraints:
            print(f"       - {c}")

        return len(constraints) >= 5

    except Exception as e:
        print(f"  [FAIL] CSP verification failed: {e}")
        return False


def verify_documentation():
    """Check documentation completeness"""
    print("\n" + "="*60)
    print("STEP 5: Verifying Documentation")
    print("="*60)

    readme_content = Path("README.md").read_text(encoding='utf-8')

    checks = {
        "CSP Formulation section": "## 🧩 CSP Formulation" in readme_content,
        "f(X) definition": "f(X) = argmax" in readme_content,
        "Defense Q&A section": "## 🎓 Defense Q&A" in readme_content,
        "Why this is AI section": "Why This Is an AI Problem" in readme_content,
        "Variables documented": "Variables:" in readme_content,
        "Constraints documented": "Constraints" in readme_content
    }

    all_ok = True
    for check_name, passed in checks.items():
        status = "[OK]" if passed else "[MISSING]"
        print(f"  {status} {check_name}")
        if not passed:
            all_ok = False

    return all_ok


def main():
    """Run all verification checks"""
    print("\n" + "#"*60)
    print("# PRE-DEFENSE VERIFICATION SCRIPT")
    print("#"*60)
    print()

    results = []

    # Run all checks
    results.append(("Files", verify_files()))
    results.append(("Imports", verify_imports()))
    results.append(("System", verify_system()))
    results.append(("CSP", verify_csp_formulation()))
    results.append(("Documentation", verify_documentation()))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(passed for _, passed in results)

    print("="*60)
    if all_passed:
        print("[PASS] ALL CHECKS PASSED - READY FOR DEFENSE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Read: QUICK_START.md")
        print("  3. Practice demo flow")
        print("\nGood luck!")
        return 0
    else:
        print("[FAIL] SOME CHECKS FAILED - PLEASE FIX BEFORE DEFENSE")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
