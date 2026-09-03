"""
Main Streamlit Web Application
AI-Powered Canteen Recommendation System for UTH
Vietnamese UI for end users, with explicit AI pipeline visualization

Uses Constraint Satisfaction Problem (CSP) + Backtracking Search + Heuristic Utility Evaluation
"""

import streamlit as st
from pathlib import Path
from knowledge_base import KnowledgeBase, MealTime, DietType, Category
from csp import UserRequest, CSPModel
from inference_engine import InferenceEngine, format_complete_trace
from utility import UtilityFunction


# Page configuration
st.set_page_config(
    page_title="Hệ thống AI gợi ý món ăn - UTH",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: clamp(1.75rem, 4vw, 2.25rem);
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: clamp(0.9rem, 2.5vw, 1.1rem);
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .menu-section {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid #CBD5E1;
    }
    .menu-section h3 {
        color: #1E293B;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .menu-image-container {
        max-width: 500px;
        margin: 0 auto;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .menu-image-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    }
    .menu-image-container img {
        width: 100%;
        height: auto;
        display: block;
    }
    .combo-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        color: #1E293B;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    .combo-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    .combo-card h3 {
        margin-top: 0;
        letter-spacing: -0.01em;
        color: #1E293B;
    }
    .score-badge {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: #FFFFFF;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .price-badge {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: #FFFFFF;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .reasoning-box {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 8px;
        color: #334155;
    }
    .pipeline-step {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3B82F6;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        color: #1E293B;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .pipeline-step h4 {
        color: #3B82F6;
        margin-top: 0;
    }
    .pipeline-step ul {
        color: #475569;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 999px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #CBD5E1, transparent);
        margin: 2rem 0;
    }
    .info-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        color: #334155;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #F1F5F9;
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #64748B;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        color: #3B82F6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    """Initialize AI system components"""
    if 'kb' not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    if 'inference_engine' not in st.session_state:
        st.session_state.inference_engine = InferenceEngine(st.session_state.kb)
    if 'csp_model' not in st.session_state:
        st.session_state.csp_model = CSPModel(st.session_state.kb)


def main():
    initialize_system()

    # Header
    st.markdown('<div class="main-header">🍽️ Hệ thống AI gợi ý món ăn</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Căn tin Trường Đại học Giao thông Vận tải TP.HCM</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Tìm món ăn phù hợp",
        "📋 Menu Căn tin",
        "🤖 Cách AI hoạt động",
        "⚙️ Cài đặt nâng cao"
    ])

    with tab1:
        show_recommendation_interface()

    with tab2:
        show_menu_gallery()

    with tab3:
        show_ai_explanation()

    with tab4:
        show_advanced_settings()


def show_menu_gallery():
    """Display canteen menu images in a separate tab"""
    st.header("📋 Menu Căn tin UTH")

    st.markdown("""
    <div style="text-align: center; color: #64748B; margin-bottom: 2rem;">
        Xem menu thực tế của căn tin để tham khảo các món ăn có sẵn
    </div>
    """, unsafe_allow_html=True)

    assets_path = Path(__file__).parent / "assets"

    if assets_path.exists():
        menu_food_path = assets_path / "menu_food.jpg"
        menu_drinks_path = assets_path / "menu_drinks.jpg"

        # Menu món ăn
        if menu_food_path.exists():
            st.markdown("""
            <div class="menu-section">
                <h3>🍚 Menu Món Ăn</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="menu-image-container">', unsafe_allow_html=True)
            st.image(str(menu_food_path), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

        # Menu đồ uống
        if menu_drinks_path.exists():
            st.markdown("""
            <div class="menu-section">
                <h3>🥤 Menu Đồ Uống</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="menu-image-container">', unsafe_allow_html=True)
            st.image(str(menu_drinks_path), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📷 Hình ảnh menu sẽ được cập nhật sớm")


def show_recommendation_interface():
    """Main recommendation interface (Tab 1)"""
    st.header("Nhập thông tin của bạn")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Thông tin cơ bản")

        # Budget input
        budget = st.slider(
            "💰 Ngân sách (VND)",
            min_value=20000,
            max_value=100000,
            value=50000,
            step=5000,
            help="Tổng số tiền bạn muốn chi cho bữa ăn (món ăn + đồ uống)"
        )

        # Meal time
        meal_time_display = st.radio(
            "⏰ Bữa ăn",
            options=["Sáng (trước 10:00)", "Trưa (sau 10:00)"],
            horizontal=True
        )
        meal_time = MealTime.BREAKFAST if "Sáng" in meal_time_display else MealTime.LUNCH

        # Diet preference
        diet_display = st.radio(
            "🥗 Chế độ ăn",
            options=["Mặn (thường)", "Chay"],
            horizontal=True
        )
        diet = DietType.CHAY if "Chay" in diet_display else DietType.MAN

    with col2:
        st.subheader("Sở thích & Trạng thái")

        # Hunger level
        hunger_level = st.select_slider(
            "😋 Mức độ đói",
            options=["light", "normal", "very_hungry"],
            value="normal",
            format_func=lambda x: {
                "light": "Đói ít 😌",
                "normal": "Đói bình thường 😊",
                "very_hungry": "Rất đói 😫"
            }[x]
        )

        # Thirsty
        is_thirsty = st.checkbox("💧 Tôi đang khát nước", value=True)

        # Taste preferences
        st.markdown("**🍜 Món bạn thích (tùy chọn):**")
        taste_prefs = []

        col_a, col_b = st.columns(2)
        with col_a:
            if st.checkbox("Cơm", value=False):
                taste_prefs.append("cơm")
            if st.checkbox("Mì/Bún/Phở", value=False):
                taste_prefs.append("mì/bún/phở")
            if st.checkbox("Bánh mì", value=False):
                taste_prefs.append("bánh mì")

        with col_b:
            if st.checkbox("Nước ép", value=False):
                taste_prefs.append("nước ép")
            if st.checkbox("Cà phê", value=False):
                taste_prefs.append("cà phê")
            if st.checkbox("Trà sữa", value=False):
                taste_prefs.append("trà sữa")

    st.markdown("---")

    # Search button
    if st.button("🔍 Tìm kiếm món ăn phù hợp", use_container_width=True):
        with st.spinner("AI đang tìm kiếm và phân tích..."):
            recommendations = get_recommendations(
                budget, meal_time, diet, hunger_level, is_thirsty, taste_prefs
            )

            if recommendations:
                display_recommendations(recommendations)
            else:
                st.error("😔 Không tìm thấy combo nào phù hợp với yêu cầu của bạn. Hãy thử tăng ngân sách hoặc điều chỉnh tiêu chí!")


def get_recommendations(budget, meal_time, diet, hunger_level, is_thirsty, taste_prefs):
    """
    Run the complete AI pipeline to get recommendations.

    This implements f(X) = argmax_{s ∈ C(X)} U(s, X)
    """
    inference_engine = st.session_state.inference_engine

    # Create user request X = (B, T, V, H, P, D)
    request = UserRequest(
        budget=budget,
        meal_time=meal_time,
        vegetarian=(diet == DietType.CHAY),
        hunger_level=hunger_level,
        taste_preferences=taste_prefs,
        wants_drink=True,  # Always recommend with drink
        is_thirsty=is_thirsty,
        caffeine_allowed=True  # Can be made configurable
    )

    # Run f(X): CSP + Backtracking + Utility + Ranking
    result = inference_engine.recommend(request, top_k=5)

    # Store result in session state for trace display
    st.session_state.last_result = result

    return result


def display_recommendations(result):
    """Display the recommended combos with reasoning"""
    if not result.ranked_solutions:
        st.error("😔 Không tìm thấy combo nào phù hợp với yêu cầu của bạn. Hãy thử tăng ngân sách hoặc điều chỉnh tiêu chí!")
        return

    st.success("✅ Tìm thấy các combo phù hợp!")
    st.header(f"🎯 Top {len(result.ranked_solutions)} gợi ý cho bạn")

    utility_function = st.session_state.inference_engine.utility_function

    # Food emoji mapping for visual appeal
    food_emojis = {
        "cơm": "🍚",
        "mì": "🍜",
        "bún": "🍜",
        "phở": "🍜",
        "bánh mì": "🥖",
        "default": "🍽️"
    }

    drink_emojis = {
        "nước ép": "🧃",
        "cà phê": "☕",
        "trà": "🍵",
        "sữa": "🥛",
        "nước": "💧",
        "default": "🥤"
    }

    def get_food_emoji(food_name):
        name_lower = food_name.lower()
        for key, emoji in food_emojis.items():
            if key in name_lower:
                return emoji
        return food_emojis["default"]

    def get_drink_emoji(drink_name):
        name_lower = drink_name.lower()
        for key, emoji in drink_emojis.items():
            if key in name_lower:
                return emoji
        return drink_emojis["default"]

    for idx, solution in enumerate(result.ranked_solutions, 1):
        food = solution.assignment.food
        drink = solution.assignment.drink

        food_emoji = get_food_emoji(food.name)
        drink_emoji = get_drink_emoji(drink.name)

        with st.container():
            st.markdown(f"""
            <div class="combo-card">
                <h3>#{idx} - {food_emoji} {food.name} + {drink_emoji} {drink.name}</h3>
                <p style="font-size: 1rem; margin-top: 1rem;">
                    <span class="price-badge">{int(solution.assignment.get_total_price()):,} VND</span>
                    <span style="margin: 0 1rem; color: #94A3B8;">•</span>
                    <span class="score-badge">{solution.normalized_score:.1f}/100 điểm</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Expandable reasoning section
            with st.expander("🧠 Vì sao gợi ý món này?"):
                st.markdown("### 📋 Lý do chi tiết:")
                for criterion, reason in solution.criteria_reasons.items():
                    st.markdown(f"- {reason}")

                st.markdown("---")
                st.markdown(utility_function.format_score_breakdown(solution))

    # Show complete inference trace
    st.markdown("---")
    with st.expander("🔬 Xem chi tiết suy luận AI (CSP + Backtracking + Utility)"):
        trace = format_complete_trace(result)
        st.markdown(trace)


def show_ai_explanation():
    """Explain how the AI system works (Tab 2)"""
    st.header("🤖 Cách hệ thống AI hoạt động")

    st.markdown("""
    ### 📊 Định nghĩa bài toán AI: f(X)

    Hệ thống này giải quyết bài toán **Constraint Satisfaction Problem (CSP)** kết hợp với
    **Backtracking Search** và **Heuristic Utility Evaluation** — kỹ thuật AI cổ điển,
    KHÔNG phải Machine Learning hay Deep Learning.

    #### 🎯 Công thức f(X):

    ```
    f(X) = argmax_{s ∈ C(X)} U(s, X)
    ```

    Trong đó:
    - **X** = Yêu cầu người dùng (input)
    - **C(X)** = Tập nghiệm khả thi (feasible solution set) thỏa mãn tất cả ràng buộc cứng
    - **U(s, X)** = Hàm utility đánh giá mức độ phù hợp của nghiệm s với yêu cầu X
    - **f(X)** = Nghiệm tối ưu có utility cao nhất
    """)

    col1, col2, col3 = st.columns([1, 0.1, 1])

    with col1:
        st.markdown("""
        <div class="pipeline-step">
            <h4>📥 INPUT (X)</h4>
            <p><strong>X = (B, T, V, H, P, D)</strong></p>
            <ul>
                <li><strong>B</strong>: Budget (ngân sách VND)</li>
                <li><strong>T</strong>: MealTime (sáng/trưa)</li>
                <li><strong>V</strong>: Vegetarian (chay/mặn)</li>
                <li><strong>H</strong>: Hunger level (mức đói)</li>
                <li><strong>P</strong>: Preferences (sở thích)</li>
                <li><strong>D</strong>: Drink requirement (cần đồ uống)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h2 style='text-align: center; color: #38BDF8;'>→</h2>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="pipeline-step">
            <h4>📤 OUTPUT f(X)</h4>
            <p><strong>Nghiệm tối ưu + Top-K gợi ý</strong></p>
            <ul>
                <li>f(X) = combo tốt nhất</li>
                <li>Top 5 combos xếp hạng</li>
                <li>Điểm utility U(s, X)</li>
                <li>Chi tiết suy luận (reasoning trace)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🔧 PIPELINE - Các bước xử lý của f:
    """)

    st.markdown("""
    ```
    X (Input)
        ↓
    1. Knowledge Base → lấy menu items
        ↓
    2. CSP Formulation → định nghĩa Variables, Domains, Constraints
        ↓
    3. Constraint Propagation → rút gọn Domains
        ↓
    4. Backtracking Search → tìm C(X) (tập nghiệm khả thi)
        ↓
    5. Utility Evaluation → tính U(s, X) cho mỗi s ∈ C(X)
        ↓
    6. Ranking → sắp xếp theo utility
        ↓
    f(X) = argmax U(s, X)
    ```
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🧩 Module 1: Knowledge Base (Cơ sở tri thức)

    **Kỹ thuật:** Biểu diễn tri thức dạng **Frame/Record** (structured knowledge representation)

    - Mỗi món ăn/đồ uống là một **frame** với các thuộc tính:
      - `name`: tên món
      - `price`: giá tiền
      - `category`: loại (cơm, mì/bún, bánh mì, nước ép, cà phê, trà sữa)
      - `diet`: mặn/chay
      - `meal_time`: sáng/trưa
      - `calories_estimate`: ước tính calories (cho matching mức đói)

    - **Dữ liệu thực:** Menu căn tin UTH
    """)

    with st.expander("📊 Xem thống kê Knowledge Base"):
        kb = st.session_state.kb
        total_dishes = len(kb.get_all_dishes())
        total_drinks = len(kb.get_all_drinks())

        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số món ăn", total_dishes)
        col2.metric("Tổng số đồ uống", total_drinks)
        col3.metric("Không gian tìm kiếm", f"{total_dishes * total_drinks:,} combos")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Module 2: CSP
    st.markdown("""
    ### 🧩 Module 2: Constraint Satisfaction Problem (CSP)

    **Mô hình hóa bài toán như một CSP:**

    #### Variables (Biến):
    - **Food** (món ăn)
    - **Drink** (đồ uống)

    #### Domains (Miền giá trị):
    - **D_Food** = {tất cả món ăn trong menu}
    - **D_Drink** = {tất cả đồ uống trong menu}

    #### Hard Constraints (Ràng buộc cứng):
    """)

    csp = st.session_state.csp_model
    for constraint_desc in csp.get_hard_constraints_description():
        st.markdown(f"- **{constraint_desc}**")

    st.markdown("""
    #### Constraint Propagation (Lan truyền ràng buộc):

    Trước khi tìm kiếm, hệ thống rút gọn miền giá trị bằng cách loại bỏ các giá trị
    rõ ràng vi phạm ràng buộc:
    - Loại món không đúng giờ
    - Loại món không phù hợp chế độ ăn
    - Loại món quá ngân sách
    - Loại đồ uống có caffeine (nếu cần)
    - Loại add-on không hợp lệ

    → Giảm không gian tìm kiếm, tăng hiệu quả
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Module 3: Backtracking Search
    st.markdown("""
    ### 🧩 Module 3: Backtracking Search

    **Thuật toán:** Tìm kiếm quay lui (Recursive Backtracking)

    ```python
    def backtrack(assignment):
        if assignment.is_complete():
            return assignment  # Tìm thấy nghiệm

        var = select_unassigned_variable()

        for value in domain[var]:
            if is_consistent(value, assignment):
                assignment[var] = value
                result = backtrack(assignment)
                if result != failure:
                    return result
                remove assignment[var]  # Quay lui

        return failure
    ```

    **Hoạt động:**
    1. Chọn biến chưa gán (Food trước, sau đó Drink)
    2. Thử từng giá trị trong miền
    3. Kiểm tra ràng buộc
    4. Nếu vi phạm → **backtrack** (quay lui)
    5. Nếu hợp lệ → tiếp tục gán biến tiếp theo
    6. Thu thập tất cả nghiệm khả thi → **C(X)**

    **Metrics được tracking:**
    - Nodes visited (số node đã duyệt)
    - Assignments tested (số phép gán thử)
    - Constraint checks (số lần kiểm tra ràng buộc)
    - Backtracks (số lần quay lui)
    - Solutions found (số nghiệm tìm được)
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Module 4: Utility Evaluation
    st.markdown("""
    ### 🧩 Module 4: Heuristic Utility Evaluation

    **Sau khi có C(X), đánh giá mỗi nghiệm bằng hàm utility U(s, X):**

    ```
    U(s, X) = w₁·budget_fit + w₂·preference_match +
              w₃·variety_bonus + w₄·hunger_fit + w₅·thirst_fit
    ```

    **5 tiêu chí đánh giá (soft preferences):**
    1. **budget_fit** (30%): Mức độ sử dụng tối ưu ngân sách (70-90% là tốt nhất)
    2. **preference_match** (25%): Khớp với sở thích món ăn đã chọn
    3. **variety_bonus** (20%): Cân bằng dinh dưỡng (kết hợp món ăn-đồ uống hợp lý)
    4. **hunger_fit** (15%): Phù hợp với mức đói (calories estimate)
    5. **thirst_fit** (10%): Phù hợp với mức khát

    → Mỗi nghiệm có điểm số U(s, X) từ 0-100

    → Xếp hạng theo điểm số: **f(X) = argmax U(s, X)**
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">

    ### 🎓 Tại sao đây là AI?

    #### ✅ Thuật toán AI được sử dụng:

    1. **Constraint Satisfaction Problem (CSP)** - mô hình hóa bài toán
    2. **Constraint Propagation** - rút gọn miền tìm kiếm
    3. **Backtracking Search** - tìm kiếm có hệ thống với quay lui
    4. **Knowledge-Based Rules** - suy luận dựa trên tri thức
    5. **Heuristic Evaluation** - đánh giá nghiệm bằng hàm utility
    6. **State-Space Search** - duyệt không gian trạng thái

    #### ✅ Đặc điểm AI cổ điển (Classical AI):
    - Symbolic reasoning (suy luận ký hiệu)
    - Explicit knowledge representation (biểu diễn tri thức tường minh)
    - Deterministic search (tìm kiếm xác định)
    - Explainable reasoning (giải thích được từng bước)

    #### ❌ KHÔNG sử dụng:
    - Machine Learning (không có training)
    - Deep Learning (không có neural networks)
    - Reinforcement Learning (không có agent học từ môi trường)
    - External AI APIs

    #### 📝 Có thể giải thích trên bảng:
    - Vẽ được CSP variables và domains
    - Demo được backtracking trên 1 ví dụ nhỏ
    - Giải thích công thức U(s, X) bằng tay
    - Chứng minh f(X) = argmax U(s, X)

    </div>
    """, unsafe_allow_html=True)


def show_advanced_settings():
    """Advanced settings for adjusting utility weights (Tab 3)"""
    st.header("⚙️ Cài đặt nâng cao")

    st.markdown("""
    ### 🎛️ Điều chỉnh trọng số hàm utility U(s, X)

    Thay đổi trọng số của các tiêu chí để xem AI ưu tiên khác nhau như thế nào.
    Tổng các trọng số sẽ được chuẩn hóa về 100%.
    """)

    inference_engine = st.session_state.inference_engine
    utility_function = inference_engine.utility_function

    col1, col2 = st.columns(2)

    with col1:
        price_weight = st.slider(
            "💰 Trọng số: Phù hợp giá",
            min_value=0.0,
            max_value=1.0,
            value=utility_function.weights["budget_fit"],
            step=0.05,
            help="Mức độ ưu tiên việc sử dụng tối ưu ngân sách (70-90% budget là tốt nhất)"
        )

        pref_weight = st.slider(
            "🎯 Trọng số: Khớp sở thích",
            min_value=0.0,
            max_value=1.0,
            value=utility_function.weights["preference_match"],
            step=0.05,
            help="Mức độ ưu tiên món ăn khớp với sở thích đã chọn"
        )

        variety_weight = st.slider(
            "🌈 Trọng số: Cân bằng dinh dưỡng",
            min_value=0.0,
            max_value=1.0,
            value=utility_function.weights["variety_bonus"],
            step=0.05,
            help="Mức độ ưu tiên sự cân bằng giữa món ăn và đồ uống"
        )

    with col2:
        hunger_weight = st.slider(
            "😋 Trọng số: Phù hợp mức đói",
            min_value=0.0,
            max_value=1.0,
            value=utility_function.weights["hunger_fit"],
            step=0.05,
            help="Mức độ ưu tiên món ăn phù hợp với mức đói (calories)"
        )

        thirst_weight = st.slider(
            "💧 Trọng số: Phù hợp mức khát",
            min_value=0.0,
            max_value=1.0,
            value=utility_function.weights["thirst_fit"],
            step=0.05,
            help="Mức độ ưu tiên đồ uống phù hợp với mức khát"
        )

    if st.button("💾 Lưu trọng số mới", use_container_width=True):
        utility_function.weights = {
            "budget_fit": price_weight,
            "preference_match": pref_weight,
            "variety_bonus": variety_weight,
            "hunger_fit": hunger_weight,
            "thirst_fit": thirst_weight
        }
        st.success("✅ Đã cập nhật trọng số! Thử tìm kiếm lại ở Tab 1 để xem sự khác biệt.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 📊 Công thức Utility hiện tại:

    ```
    U(s, X) = {:.2f}·budget_fit + {:.2f}·preference_match +
              {:.2f}·variety_bonus + {:.2f}·hunger_fit + {:.2f}·thirst_fit
    ```

    Tổng trọng số: **{:.0f}%** (sẽ được chuẩn hóa về 100%)
    """.format(
        price_weight * 100,
        pref_weight * 100,
        variety_weight * 100,
        hunger_weight * 100,
        thirst_weight * 100,
        (price_weight + pref_weight + variety_weight + hunger_weight + thirst_weight) * 100
    ))

    if st.button("🔄 Khôi phục trọng số mặc định"):
        utility_function.weights = {
            "budget_fit": 0.30,
            "preference_match": 0.25,
            "variety_bonus": 0.20,
            "hunger_fit": 0.15,
            "thirst_fit": 0.10
        }
        st.success("✅ Đã khôi phục trọng số mặc định!")
        st.rerun()


if __name__ == "__main__":
    main()
