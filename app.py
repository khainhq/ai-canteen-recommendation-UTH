"""Student-friendly Vietnamese UI for the UTH canteen recommendation system."""

from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st

from knowledge_base import KnowledgeBase, MealTime, DietType
from csp import UserRequest, CSPModel
from inference_engine import InferenceEngine, format_complete_trace


st.set_page_config(
    page_title="Hôm nay ăn gì? | Căn tin UTH",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ASSETS_PATH = Path(__file__).parent / "assets"
st.markdown(
    f"<style>{(ASSETS_PATH / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


def initialize_system():
    """Initialize the existing AI components once per session."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    if "inference_engine" not in st.session_state:
        st.session_state.inference_engine = InferenceEngine(st.session_state.kb)
    if "csp_model" not in st.session_state:
        st.session_state.csp_model = CSPModel(st.session_state.kb)


def main():
    initialize_system()
    st.markdown("""
    <div class="hero-header">
        <div>
            <div class="eyebrow">CĂN TIN UTH · BẠN ĐỒNG HÀNH GIỜ ĂN</div>
            <h1>Hôm nay ăn gì? 🍃</h1>
            <p>Bớt phân vân, thêm ngon miệng. Chọn combo hợp gu, vừa túi tiền sinh viên.</p>
        </div>
        <div class="hero-sticker" aria-hidden="true">🍱</div>
    </div>
    """, unsafe_allow_html=True)

    find_tab, menu_tab, help_tab = st.tabs([
        "🍽️ Chọn món", "📋 Menu căn tin", "🌷 Hướng dẫn",
    ])
    with find_tab:
        show_recommendation_interface()
    with menu_tab:
        show_menu_gallery()
    with help_tab:
        show_usage_guide()
        with st.expander("🤖 Tìm hiểu cách AI gợi ý món"):
            show_ai_explanation()
        with st.expander("⚙️ Nâng cao: điều chỉnh tiêu chí ưu tiên"):
            show_advanced_settings()
    st.markdown(
        '<p class="footer-note">Dành cho sinh viên UTH · Giá và món thực tế có thể thay đổi tại căn tin.</p>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def menu_image_data(path_string, modified_time):
    """Cache image encoding; invalidate it when a menu photo changes."""
    return b64encode(Path(path_string).read_bytes()).decode("ascii")


def show_menu_gallery():
    st.header("Menu nhỏ, lựa chọn nhiều 🍚")
    st.caption("Hai bảng menu để bạn tham khảo. Mở mục “Xem rõ menu” bên dưới ảnh để phóng to.")
    for column, filename, title in zip(
        st.columns(2),
        ["menu_food.jpg", "menu_drinks.jpg"],
        ["🍚 Món ăn", "🥤 Đồ uống"],
    ):
        with column:
            st.subheader(title)
            image_path = ASSETS_PATH / filename
            if not image_path.is_file():
                st.info("Ảnh menu này đang được cập nhật.")
                continue
            image_data = menu_image_data(str(image_path), image_path.stat().st_mtime_ns)
            # A single HTML figure actually contains the image, unlike HTML wrappers
            # opened and closed around separate Streamlit elements.
            st.markdown(f"""
            <figure class="menu-preview">
                <img src="data:image/jpeg;base64,{image_data}" alt="Bảng giá {title} căn tin UTH">
                <figcaption>Bản xem trước · Mở ảnh đầy đủ bên dưới</figcaption>
            </figure>
            """, unsafe_allow_html=True)
            with st.expander(f"🔎 Xem rõ menu {title.split(' ', 1)[1].lower()}"):
                st.image(str(image_path), use_container_width=True)
                st.caption("Dùng nút toàn màn hình trên ảnh để phóng to.")


def show_usage_guide():
    st.header("Một bữa ngon chỉ với 3 bước 🌷")
    st.markdown("""
    <div class="guide-grid">
        <article class="guide-card">
            <div class="step">BƯỚC 01 · CHỌN NHU CẦU</div>
            <h3>💰 Hôm nay chi bao nhiêu?</h3>
            <p>Ở mục Chọn món, đặt ngân sách cho cả món ăn và đồ uống. Chọn bữa sáng hoặc trưa, rồi chọn ăn mặn hay chay.</p>
        </article>
        <article class="guide-card">
            <div class="step">BƯỚC 02 · THÊM CHÚT GU</div>
            <h3>😋 Bạn đang thèm gì?</h3>
            <p>Chọn mức độ đói, đánh dấu nếu đang khát và thêm nhóm món yêu thích. Chưa biết ăn gì? Cứ để trống sở thích nhé.</p>
        </article>
        <article class="guide-card">
            <div class="step">BƯỚC 03 · NHẬN GỢI Ý</div>
            <h3>🍱 Chốt combo hợp mình</h3>
            <p>Nhấn “Gợi ý bữa ăn cho mình”. So sánh giá, điểm phù hợp và mở phần lý do để hiểu vì sao combo được đề xuất.</p>
        </article>
    </div>
    """, unsafe_allow_html=True)
    st.info("💡 Thử nhanh: ngân sách 35.000đ → bữa trưa → ăn mặn → mức đói bình thường → chọn Cơm → nhận gợi ý.")
    with st.expander("Chưa tìm được combo phù hợp thì sao?"):
        st.write("Thử tăng ngân sách từng 5.000đ hoặc kiểm tra lại bữa ăn đã chọn. Hệ thống chỉ gợi ý combo đáp ứng ngân sách, thời điểm và chế độ ăn; không tự bỏ qua các điều kiện này.")
    with st.expander("Điểm phù hợp có ý nghĩa gì?"):
        st.write("Điểm dùng để so sánh các combo theo giá, sở thích, sự kết hợp món, mức đói và mức khát. Điểm cao hơn là phù hợp hơn với tiêu chí đã chọn, không phải đánh giá chất lượng món hay tư vấn dinh dưỡng.")
        st.caption("Mỗi combo gồm một món ăn và một đồ uống. Sở thích là tiêu chí ưu tiên, không phải điều kiện bắt buộc.")
    with st.expander("Làm sao xem menu và đổi lựa chọn?"):
        st.write("Vào Menu căn tin và mở “Xem rõ menu” bên dưới mỗi ảnh. Để đổi combo, quay lại Chọn món, sửa thông tin và nhấn nút gợi ý lần nữa. Kết quả cũ vẫn được giữ cho đến lần tìm tiếp theo.")
        st.caption("Ứng dụng chỉ hỗ trợ gợi ý, chưa có chức năng đặt món hoặc thanh toán. Nếu có dị ứng thực phẩm, hãy xác nhận thành phần trực tiếp với căn tin.")


def show_recommendation_interface():
    st.markdown("""
    <div class="quick-guide">
        <span><b>01</b> Chọn ngân sách</span>
        <span><b>02</b> Thêm sở thích</span>
        <span><b>03</b> Nhận combo hợp gu</span>
    </div>
    """, unsafe_allow_html=True)
    with st.form("meal_preferences"):
        st.subheader("Một chút thông tin, một bữa đúng ý ✨")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            budget = st.slider(
                "💰 Ngân sách cho cả bữa (đ)", min_value=20000,
                max_value=100000, value=50000, step=5000,
                help="Tổng tiền cho một món ăn và một đồ uống.",
            )
            meal_time_display = st.radio(
                "⏰ Bạn ăn bữa nào?", ["Sáng (trước 10:00)", "Trưa (sau 10:00)"],
                horizontal=True,
            )
            diet_display = st.radio("🥗 Chế độ ăn", ["Mặn (thường)", "Chay"], horizontal=True)
        with col2:
            hunger_level = st.select_slider(
                "😋 Bạn đói đến đâu?", options=["light", "normal", "very_hungry"],
                value="normal", format_func=lambda value: {
                    "light": "Đói nhẹ", "normal": "Vừa vừa", "very_hungry": "Đói lắm",
                }[value],
            )
            is_thirsty = st.checkbox("💧 Mình đang khát nước", value=True)
            taste_labels = st.multiselect(
                "🍜 Món bạn thích (không bắt buộc)",
                ["Cơm", "Mì/Bún/Phở", "Bánh mì", "Nước ép", "Cà phê", "Trà sữa"],
                placeholder="Chọn món bạn thích…",
                help="Có thể chọn nhiều nhóm món hoặc để trống để khám phá.",
            )
        st.caption("🍃 Mỗi gợi ý gồm món ăn + đồ uống, trong ngân sách bạn chọn.")
        submitted = st.form_submit_button("✨ Gợi ý bữa ăn cho mình", use_container_width=True)

    if submitted:
        meal_time = MealTime.BREAKFAST if "Sáng" in meal_time_display else MealTime.LUNCH
        diet = DietType.CHAY if diet_display == "Chay" else DietType.MAN
        with st.spinner("Đang tìm bữa ăn hợp gu của bạn..."):
            get_recommendations(
                budget, meal_time, diet, hunger_level, is_thirsty,
                [label.lower() for label in taste_labels],
            )
        st.session_state.result_summary = (
            f"Ngân sách {budget:,.0f}đ · {meal_time_display} · {diet_display}"
        )
    if "last_result" in st.session_state:
        st.caption("Lần tìm gần nhất: " + st.session_state.get("result_summary", ""))
        display_recommendations(st.session_state.last_result)
    else:
        st.caption("Chưa biết bắt đầu từ đâu? Xem mục 🌷 Hướng dẫn để thử một ví dụ nhé.")


def get_recommendations(budget, meal_time, diet, hunger_level, is_thirsty, taste_prefs):
    """Run the original CSP, backtracking, utility and ranking pipeline."""
    request = UserRequest(
        budget=budget, meal_time=meal_time, vegetarian=(diet == DietType.CHAY),
        hunger_level=hunger_level, taste_preferences=taste_prefs,
        wants_drink=True, is_thirsty=is_thirsty, caffeine_allowed=True,
    )
    result = st.session_state.inference_engine.recommend(request, top_k=5)
    st.session_state.last_result = result
    return result


def display_recommendations(result):
    if not result.ranked_solutions:
        st.warning("Chưa có combo phù hợp 🌱 Thử tăng ngân sách hoặc kiểm tra lại bữa ăn và chế độ ăn nhé.")
        return
    st.subheader(f"Dành cho bạn · {len(result.ranked_solutions)} combo hợp gu")
    utility_function = st.session_state.inference_engine.utility_function
    for idx, solution in enumerate(result.ranked_solutions, 1):
        food = solution.assignment.food
        drink = solution.assignment.drink
        rank_label = "🌟 Gợi ý phù hợp nhất" if idx == 1 else f"Lựa chọn {idx:02d}"
        card_class = "combo-card best" if idx == 1 else "combo-card"
        st.markdown(f"""
        <article class="{card_class}">
            <div class="combo-rank">{rank_label}</div>
            <h3>{escape(food.name)} + {escape(drink.name)}</h3>
            <div class="combo-meta">
                <span class="price-badge">{int(solution.assignment.get_total_price()):,}đ / combo</span>
                <span class="score-badge">★ {solution.normalized_score:.1f}/100 điểm phù hợp</span>
            </div>
        </article>
        """, unsafe_allow_html=True)
        with st.expander(f"Vì sao chọn combo {idx}?"):
            for reason in solution.criteria_reasons.values():
                st.markdown(f"- {reason}")
            st.markdown(utility_function.format_score_breakdown(solution))
    with st.expander("🔬 Chi tiết kỹ thuật của lần gợi ý này"):
        st.markdown(format_complete_trace(result))


def show_ai_explanation():
    """Keep the educational explanation available without crowding the main UI."""
    st.markdown("""
    **AI chọn món như thế nào?**

    1. **Đọc menu:** lấy món ăn, đồ uống, giá và thông tin từ cơ sở tri thức.
    2. **Lọc điều kiện bắt buộc (CSP):** ngân sách, bữa ăn, chế độ ăn và các ràng buộc của hệ thống.
    3. **Tìm combo bằng Backtracking:** thử các cách kết hợp hợp lệ, quay lui khi vi phạm điều kiện.
    4. **Chấm điểm Utility:** đánh giá giá tiền, sở thích, sự kết hợp món, mức đói và mức khát.
    5. **Xếp hạng:** trả về tối đa 5 combo có điểm phù hợp cao nhất.

    Đây là **AI dựa trên tri thức và tìm kiếm**, không phải mô hình học máy hay dịch vụ AI bên ngoài.
    Mở phần chi tiết kỹ thuật dưới kết quả để xem dấu vết suy luận thực tế.
    """)
    st.code("f(X) = argmax_{s ∈ C(X)} U(s, X)", language=None)
    st.caption("X: yêu cầu của bạn · C(X): các combo hợp lệ · U(s, X): điểm phù hợp của một combo.")
    kb = st.session_state.kb
    food_col, drink_col = st.columns(2)
    food_col.metric("Món ăn trong dữ liệu", len(kb.get_all_dishes()))
    drink_col.metric("Đồ uống trong dữ liệu", len(kb.get_all_drinks()))
    st.markdown("**Các điều kiện bắt buộc**")
    for description in st.session_state.csp_model.get_hard_constraints_description():
        st.markdown(f"- {description}")


DEFAULT_WEIGHTS = {
    "budget_fit": 0.30, "preference_match": 0.25, "variety_bonus": 0.20,
    "hunger_fit": 0.15, "thirst_fit": 0.10,
}


def reset_weights():
    st.session_state.inference_engine.utility_function.weights = DEFAULT_WEIGHTS.copy()
    for key, value in DEFAULT_WEIGHTS.items():
        st.session_state[f"weight_{key}"] = round(value * 100)
    st.session_state.pop("last_result", None)


def show_advanced_settings():
    st.caption("Không cần chỉnh để sử dụng. Tăng tiêu chí bạn muốn ưu tiên; khi lưu, tổng sẽ được chuẩn hóa về 100%.")
    utility_function = st.session_state.inference_engine.utility_function
    labels = {
        "budget_fit": "💰 Phù hợp giá", "preference_match": "🎯 Hợp sở thích",
        "variety_bonus": "🌈 Kết hợp món", "hunger_fit": "😋 Phù hợp mức đói",
        "thirst_fit": "💧 Phù hợp mức khát",
    }
    weights = {}
    for key, label in labels.items():
        widget_key = f"weight_{key}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = round(utility_function.weights[key] * 100)
        weights[key] = st.slider(label, 0, 100, step=5, key=widget_key)
    if st.button("Lưu mức ưu tiên", type="primary"):
        total = sum(weights.values())
        if total == 0:
            st.error("Hãy đặt ít nhất một tiêu chí lớn hơn 0 trước khi lưu.")
        else:
            utility_function.weights = {key: value / total for key, value in weights.items()}
            st.session_state.pop("last_result", None)
            st.session_state.weights_saved = True
            st.rerun()
    if st.session_state.pop("weights_saved", False):
        st.success("Đã lưu! Quay lại Chọn món và tìm lại để áp dụng mức ưu tiên mới.")
    st.button("Khôi phục mặc định", on_click=reset_weights)


if __name__ == "__main__":
    main()
