"""Streamlit interface for the Vietnamese BLIP image assistant."""

from __future__ import annotations

import hashlib
import io
import re
import time
from dataclasses import replace

import streamlit as st
from PIL import Image

from src.alignment import ClipAligner
from src.captioner import BlipCaptioner
from src.confidence import ConfidenceAssessor, ConfidenceEvidence, load_policy
from src.model_runtime import GenerationConfig, GenerationEvidence
from src.vqa import BlipVQA


st.set_page_config(
    page_title="Thị Giác · Trợ lý ảnh",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
:root { --ink:#18332d; --muted:#687a73; --paper:#f5f5ef; --surface:#fffefa; --line:#dfe6dc; --green:#1e6a53; --mint:#e7f1e9; --gold:#e8b65a; }
.stApp { background: radial-gradient(ellipse at 92% 0%, #e8eee4 0, transparent 29rem), var(--paper); color:var(--ink); }
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"] {
  font-family:'Be Vietnam Pro', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#eef1e9; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
.block-container { max-width:1450px; padding-top:2.1rem; padding-bottom:3rem; }
.masthead { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; padding:0 0 1.35rem; border-bottom:1px solid #d9e0d5; margin-bottom:1.6rem; }
.eyebrow { color:var(--green); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.masthead h1 { font-family:'Be Vietnam Pro','Segoe UI',sans-serif; font-size:clamp(2rem,4vw,3rem); font-weight:700; line-height:1.18; letter-spacing:-.035em; text-wrap:balance; margin:.35rem 0 .55rem; color:var(--ink); }
.masthead p { max-width:45rem; margin:0; color:var(--muted); line-height:1.65; font-size:.95rem; }
.status-chip { background:#e5f1e9; color:#28684f; border:1px solid #cfe2d3; border-radius:999px; padding:.46rem .78rem; white-space:nowrap; font-size:.76rem; font-weight:700; letter-spacing:.04em; }
.section-kicker { color:#75847c; text-transform:uppercase; letter-spacing:.13em; font-size:.7rem; font-weight:700; margin:.25rem 0 .55rem; }
.welcome-panel { min-height:17rem; padding:2.25rem; border:1px solid var(--line); border-radius:1.2rem; background:linear-gradient(140deg,#fffefa 0%,#f0f4ed 100%); display:flex; flex-direction:column; justify-content:center; }
.welcome-panel h2 { color:var(--ink); font-family:'Be Vietnam Pro','Segoe UI',sans-serif; font-size:1.85rem; font-weight:650; line-height:1.3; letter-spacing:-.025em; margin:.1rem 0 .55rem; }
.welcome-panel p { color:var(--muted); max-width:40rem; line-height:1.75; }
.welcome-mark { width:2.8rem; height:2.8rem; display:grid; place-items:center; border-radius:.9rem; background:#e4eee4; color:var(--green); font-size:1.25rem; }
.stButton > button { border-radius:.78rem; min-height:2.8rem; font-weight:600; transition:transform .16s ease, box-shadow .16s ease; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 5px 14px #18332d1c; }
.stButton > button[kind="primary"] { background:var(--green); border-color:var(--green); }
[data-testid="stChatMessage"] { border:1px solid #e2e8df; background:#fffefa; border-radius:1rem; padding:.85rem 1rem; }
[data-testid="stChatMessage"] p { font-family:'Be Vietnam Pro','Segoe UI',sans-serif; line-height:1.72; letter-spacing:-.008em; }
[data-testid="stChatInput"] { background:#fffefa; border-color:#d8e1d7; border-radius:1rem; }
[data-testid="stFileUploader"] section { background:#fbfcf8; border:1px dashed #a9b9a8; border-radius:1rem; }
[data-testid="stMetric"] { background:#fffefa; border:1px solid var(--line); padding:.8rem 1rem; border-radius:.9rem; }
div[data-testid="stExpander"] { background:#f8faf5; border:1px solid var(--line); border-radius:.9rem; }
hr { border-color:#e1e6de; }
@media (max-width:760px) { .block-container{padding:1rem 1rem 2rem}.masthead{align-items:flex-start;flex-direction:column;gap:.8rem}.masthead h1{font-size:2.2rem}.welcome-panel{padding:1.4rem;min-height:13rem} }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


_LEGACY_REVIEW_NOTE = re.compile(
    r"\n{2}\*Câu (?:mô tả|trả lời)[^*]*?(?:đối chiếu|kiểm tra)[^*]*?ảnh\.\*"
)


def _remove_legacy_review_note(text: str) -> str:
    return _LEGACY_REVIEW_NOTE.sub("", text)


@st.cache_resource(show_spinner="Đang nạp mô hình caption…")
def get_captioner() -> BlipCaptioner:
    return BlipCaptioner()


@st.cache_resource(show_spinner="Đang nạp mô hình hỏi đáp…")
def get_vqa() -> BlipVQA:
    return BlipVQA()


@st.cache_resource(show_spinner=False)
def get_assessor() -> ConfidenceAssessor:
    return ConfidenceAssessor(load_policy())


@st.cache_resource(show_spinner=False)
def get_aligner() -> ClipAligner:
    return ClipAligner()


def _init_state() -> None:
    defaults = {
        "messages": [],
        "records": [],
        "image_fingerprint": None,
        "last_response": "",
        "last_audio": None,
        "last_decision": None,
        "pending_question": "",
        "transcript_text": "",
        "notice": "",
    }
    for name, value in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = value
    for message in st.session_state.messages:
        message["content"] = _remove_legacy_review_note(message["content"])
    st.session_state.last_response = _remove_legacy_review_note(st.session_state.last_response)


def _generation_config(
    beams: int, tokens: int, temperature: float, sampling: bool, check_confidence: bool
) -> GenerationConfig:
    generation = GenerationConfig(
        num_beams=beams,
        max_new_tokens=tokens,
        temperature=temperature if sampling else 1.0,
        do_sample=sampling,
    )
    if check_confidence and (generation.num_beams < 2 or generation.do_sample):
        generation = replace(generation, num_beams=3, do_sample=False, temperature=1.0)
    return generation


def _confidence(result: GenerationEvidence, image: Image.Image, use_clip: bool):
    clip_similarity = None
    warning = None
    if use_clip:
        try:
            clip_similarity = get_aligner().similarity(image, result.text)
        except Exception as exc:  # Keep the answer usable when optional CLIP is unavailable.
            warning = f"Đối chiếu CLIP không khả dụng: {exc}"
    evidence = ConfidenceEvidence(
        mean_logprob=result.mean_logprob,
        beam_margin=result.beam_margin,
        primary_text=result.text,
        alternative_text=result.alternative_text,
        clip_similarity=clip_similarity,
    )
    return get_assessor().assess(evidence), warning


def _append_assistant(text: str, kind: str = "answer") -> None:
    st.session_state.messages.append({"role": "assistant", "content": text, "kind": kind})
    st.session_state.last_response = text


def _speak_latest(language: str) -> None:
    if not st.session_state.last_response:
        st.session_state.notice = "Chưa có câu trả lời để đọc thành tiếng."
        return
    try:
        from src.speech import synthesize_speech

        st.session_state.last_audio = synthesize_speech(st.session_state.last_response, language)
        st.session_state.notice = "Đã tạo giọng đọc."
    except Exception as exc:
        st.session_state.notice = f"Không thể tạo giọng đọc: {exc}"


_init_state()

with st.sidebar:
    workspace_title, guide_menu = st.columns([5, 1], vertical_alignment="center")
    with workspace_title:
        st.markdown('<div class="section-kicker">Không gian làm việc</div>', unsafe_allow_html=True)
    with guide_menu:
        with st.popover("⋮", type="tertiary", help="Mở hướng dẫn sử dụng"):
            st.markdown("#### Hướng dẫn sử dụng")
            st.markdown(
                """
                **Ảnh và hỏi đáp**  
                Tải ảnh, tạo mô tả hoặc nhập câu hỏi tiếng Anh ở khung hội thoại.

                **Nhận dạng giọng nói**  
                Mở app bằng `http://localhost:8501`. Nếu micro bị chặn, mở quyền trang web ở bên trái thanh địa chỉ, chọn **Microphone → Cho phép**, rồi tải lại trang.

                **Giọng đọc**  
                Chọn ngôn ngữ TTS và bấm **Đọc câu trả lời gần nhất** sau khi AI đã trả lời.

                **Mô hình**  
                Caption và VQA dùng BLIP; STT tiếng Việt dùng Gipformer 1.5.
                """
            )
    st.markdown("### 01 · Chọn ảnh")
    uploaded_image = st.file_uploader(
        "Tải ảnh cần phân tích", type=["png", "jpg", "jpeg", "webp"],
        help="Ảnh được xử lý trong ứng dụng để tạo mô tả và trả lời câu hỏi.",
    )
    image = None
    image_fingerprint = None
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        image_fingerprint = hashlib.sha256(image_bytes).hexdigest()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption=uploaded_image.name, use_container_width=True)
        except Exception as exc:
            st.error(f"Không đọc được ảnh này: {exc}")
            image = None

    if image_fingerprint != st.session_state.image_fingerprint:
        st.session_state.image_fingerprint = image_fingerprint
        st.session_state.messages = []
        st.session_state.records = []
        st.session_state.last_response = ""
        st.session_state.last_audio = None
        st.session_state.last_decision = None
        st.session_state.notice = "Ảnh đã được cập nhật. Hãy bắt đầu một cuộc hội thoại mới."

    caption_button = st.button(
        "Tạo mô tả ảnh", type="primary", use_container_width=True, disabled=image is None,
    )
    new_chat_button = st.button("Xóa hội thoại", use_container_width=True)
    if new_chat_button:
        st.session_state.messages = []
        st.session_state.records = []
        st.session_state.last_response = ""
        st.session_state.last_audio = None
        st.session_state.last_decision = None
        st.session_state.notice = "Đã bắt đầu hội thoại mới."

    st.markdown("---")
    st.markdown("### 02 · Cấu hình sinh")
    with st.expander("Tham số mô hình", expanded=False):
        beams = st.slider("Số beam", min_value=1, max_value=5, value=1, help="Tăng beam để so sánh nhiều phương án sinh.")
        tokens = st.slider("Độ dài tối đa", min_value=5, max_value=60, value=30, help="Số token tối đa trong mô tả hoặc câu trả lời.")
        sampling = st.checkbox("Bật sampling", value=False)
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, disabled=not sampling)
        check_confidence = st.checkbox("Đánh giá bằng chứng và cho phép từ chối", value=True)
        if check_confidence:
            st.caption("Chế độ này dùng beam search xác định với 3 beam để thu thập thêm tín hiệu bằng chứng.")
        use_clip = st.checkbox("Đối chiếu ảnh bằng CLIP", value=False, help="Tải thêm CLIP khi bật lần đầu.")
    st.markdown("---")
    st.markdown("### 03 · Nhận dạng giọng nói · STT")
    recorded_audio = st.audio_input(
        "Ghi âm câu hỏi",
        key="speech_recording",
        help="Trình duyệt sẽ hỏi quyền micro ở lần sử dụng đầu tiên. Không mở app bằng địa chỉ 192.168.x.x.",
    )
    uploaded_audio = st.file_uploader(
        "Hoặc tải bản ghi âm",
        type=["wav", "mp3", "m4a", "webm"],
        key="speech_upload",
    )
    audio_input = recorded_audio or uploaded_audio
    transcribe_clicked = st.button(
        "Chuyển giọng nói thành văn bản",
        key="transcribe_audio",
        type="primary",
        use_container_width=True,
        disabled=audio_input is None,
    )
    if transcribe_clicked and audio_input is not None:
        try:
            from src.speech import transcribe_audio

            with st.spinner("Đang nhận dạng tiếng Việt…"):
                st.session_state.transcript_text = transcribe_audio(audio_input.getvalue(), "vi")
            if not st.session_state.transcript_text:
                st.warning("Chưa nhận dạng được lời nói. Hãy thử ghi âm rõ hơn.")
        except Exception as exc:
            st.error(f"Không thể chuyển giọng nói thành văn bản: {exc}")
    transcript = st.text_area(
        "Văn bản nhận dạng",
        key="transcript_text",
        height=100,
        help="Bạn có thể sửa nội dung trước khi gửi vào hội thoại.",
    )
    use_transcript = st.button(
        "Đưa câu vào hội thoại",
        key="use_transcript",
        use_container_width=True,
        disabled=not transcript.strip() or image is None,
        help="Tải ảnh lên trước để gửi câu hỏi về ảnh.",
    )
    st.markdown("---")
    st.markdown("### 04 · Giọng đọc câu trả lời · TTS")
    speech_language = st.selectbox("Ngôn ngữ giọng đọc TTS", [("Tiếng Việt · giọng nữ miền Bắc", "vi"), ("English", "en")], format_func=lambda option: option[0], index=0)
    voice_code = speech_language[1]
    auto_speak = st.checkbox("Tự đọc câu trả lời", value=False)
    if st.button("Đọc câu trả lời gần nhất", use_container_width=True, disabled=not st.session_state.last_response):
        _speak_latest(voice_code)
    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format="audio/wav")


st.markdown(
    '<header class="masthead"><div><div class="eyebrow">Thị Giác · phòng thí nghiệm hình ảnh</div>'
    '<h1>Nhìn kỹ hơn. Hỏi tự nhiên hơn.</h1>'
    '<p>Tạo mô tả, đặt câu hỏi tiếng Anh về nội dung trong ảnh và theo dõi câu trả lời trong cùng một cuộc hội thoại.</p>'
    '</div><div class="status-chip">BLIP · ENGLISH VQA</div></header>',
    unsafe_allow_html=True,
)

generation = _generation_config(beams, tokens, temperature, sampling, check_confidence)

if caption_button and image is not None:
    try:
        with st.spinner("Đang quan sát các chi tiết trong ảnh…"):
            started = time.perf_counter()
            captioner = get_captioner()
            result = captioner.caption_with_evidence(image, generation)
            latency = (time.perf_counter() - started) * 1000
            decision, confidence_warning = _confidence(result, image, use_clip) if check_confidence else (None, None)
            display = result.text
            if decision and decision.should_abstain:
                display = "Tôi chưa thể tạo mô tả đủ tin cậy từ ảnh này."
            _append_assistant(f"**Mô tả ảnh**\n\n{display}", "caption")
            st.session_state.last_response = display
            st.session_state.last_audio = None
            st.session_state.last_decision = decision
            st.session_state.records.append({"loại": "Caption", "nội dung": display, "thời gian": latency})
            st.session_state.notice = f"Đã tạo mô tả · {latency:.0f} ms" + (f" · {confidence_warning}" if confidence_warning else "")
            if auto_speak:
                _speak_latest(voice_code)
    except Exception as exc:
        st.error(f"Không thể tạo mô tả: {exc}")

if use_transcript:
    st.session_state.pending_question = transcript.strip()

pending_prompt = st.session_state.pending_question
st.session_state.pending_question = ""
typed_prompt = st.chat_input("Ask in English about objects, colors, positions, or actions…", disabled=image is None)
prompt = pending_prompt or typed_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if image is None:
        _append_assistant("Bạn hãy tải ảnh lên trước để mình có thể trả lời.")
    else:
        try:
            with st.spinner("Đang tìm câu trả lời trong ảnh…"):
                started = time.perf_counter()
                result = get_vqa().answer_with_evidence(image, prompt, generation)
                latency = (time.perf_counter() - started) * 1000
                decision, confidence_warning = _confidence(result, image, use_clip) if check_confidence else (None, None)
                display = result.text
                if decision and decision.should_abstain:
                    display = "Tôi chưa thể xác định điều này từ ảnh với đủ bằng chứng."
                _append_assistant(display)
                st.session_state.last_audio = None
                st.session_state.last_decision = decision
                st.session_state.records.append({"loại": "VQA", "câu hỏi": prompt, "câu trả lời": result.text, "thời gian": latency})
                st.session_state.notice = f"Đã trả lời · {latency:.0f} ms" + (f" · {confidence_warning}" if confidence_warning else "")
                if auto_speak:
                    _speak_latest(voice_code)
        except Exception as exc:
            _append_assistant(f"Không thể trả lời câu hỏi này: {exc}")
            st.session_state.notice = "Xử lý câu hỏi chưa thành công."

left, right = st.columns([1.65, 0.85], gap="large")
with left:
    st.markdown('<div class="section-kicker">Cuộc hội thoại</div>', unsafe_allow_html=True)
    if not image:
        st.markdown(
            '<section class="welcome-panel"><div class="welcome-mark">◉</div>'
            '<h2>Bắt đầu từ một khung hình</h2>'
            '<p>Tải một ảnh lên để bắt đầu phân tích.</p>'
            '</section>',
            unsafe_allow_html=True,
        )
    elif not st.session_state.messages:
        st.markdown(
            '<section class="welcome-panel"><div class="welcome-mark">01</div>'
            '<h2>Ảnh đã sẵn sàng</h2>'
            '<p>Tạo mô tả hoặc đặt câu hỏi bằng tiếng Anh trong khung hội thoại.</p>'
            '</section>',
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if image and st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format="audio/wav")

with right:
    st.markdown('<div class="section-kicker">Thông tin phiên</div>', unsafe_allow_html=True)
    st.metric("Lượt phân tích", len(st.session_state.records))
    if st.session_state.records:
        latest = st.session_state.records[-1]
        st.metric("Độ trễ gần nhất", f"{latest['thời gian']:.0f} ms")
    else:
        st.metric("Độ trễ gần nhất", "—")
    st.markdown("#### Bằng chứng")
    decision = st.session_state.last_decision
    if decision is None:
        st.caption("Chưa có dữ liệu đánh giá.")
    else:
        label = {"trusted": "Tín hiệu nhất quán", "review": "Cần đối chiếu", "abstain": "Chưa đủ bằng chứng"}[decision.level]
        st.write(label)
        st.caption("Điểm dưới đây là chỉ số nội bộ, chưa phải xác suất đã hiệu chỉnh.")
        if decision.signals:
            for signal, value in decision.signals.items():
                st.progress(float(min(1, max(0, value))), text=signal.replace("_", " "))
    if st.session_state.notice:
        st.caption(st.session_state.notice)
    st.markdown("---")
    st.markdown("#### Lịch sử lượt hỏi")
    if st.session_state.records:
        for index, record in enumerate(reversed(st.session_state.records), start=1):
            with st.expander(f"{record['loại']} · {record['thời gian']:.0f} ms · lượt {len(st.session_state.records) - index + 1}"):
                if "câu hỏi" in record:
                    st.markdown(f"**Hỏi:** {record['câu hỏi']}")
                    st.markdown(f"**Đáp:** {record['câu trả lời']}")
                else:
                    st.markdown(record["nội dung"])
    else:
        st.caption("Lịch sử sẽ xuất hiện tại đây sau lượt phân tích đầu tiên.")

