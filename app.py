"""Modern Gradio application for image captioning and visual Q&A."""
import time
from dataclasses import replace

from src.alignment import ClipAligner
from src.captioner import BlipCaptioner
from src.confidence import ConfidenceAssessor, ConfidenceEvidence, load_policy, render_confidence
from src.model_runtime import GenerationConfig, GenerationEvidence
from src.session import ConversationSession
from src.speech import SpeechService
from src.vqa import BlipVQA


APP_CSS = """
:root { --brand: #6d5dfc; --brand-dark: #26245d; --mint: #22c5a5; }
.gradio-container { max-width: 1380px !important; margin: 0 auto !important; }
.app-shell { padding: .75rem 0 1rem; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: .8rem 1.1rem; margin-bottom: .8rem; border: 1px solid rgba(100,116,139,.16);
  border-radius: 18px; background: rgba(255,255,255,.84); box-shadow: 0 8px 28px rgba(15,23,42,.06); }
.brand { display: flex; align-items: center; gap: .75rem; }
.brand-mark { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 14px;
  color: white; font-size: 1.2rem; background: linear-gradient(135deg,var(--brand),var(--mint)); }
.brand h1 { margin: 0; font-size: 1.12rem; line-height: 1.2; }
.brand p { margin: .15rem 0 0; color: #64748b; font-size: .82rem; }
.online { padding: .35rem .65rem; border-radius: 999px; color: #13715e; background: #e7fbf5;
  border: 1px solid #b8efdf; font-size: .78rem; font-weight: 700; }
.sidebar, .chat-panel { border: 1px solid rgba(100,116,139,.18) !important; border-radius: 20px !important;
  box-shadow: 0 12px 36px rgba(15,23,42,.07) !important; background: rgba(255,255,255,.8) !important; }
.sidebar { padding: .7rem !important; }
.sidebar-title h3 { margin: 0 0 .15rem; }
.sidebar-title p { margin: 0 0 .65rem; color: #64748b; font-size: .86rem; }
.chat-panel { padding: .35rem .55rem .65rem !important; }
.chat-heading { padding: .55rem .45rem .3rem; }
.chat-heading h2 { margin: 0; font-size: 1.08rem; }
.chat-heading p { margin: .15rem 0 0; color: #64748b; font-size: .84rem; }
.chat-window { border: 0 !important; background: transparent !important; }
.composer { padding: .45rem !important; border-radius: 16px !important; background: #f7f8fc !important;
  border: 1px solid rgba(100,116,139,.16) !important; }
.send-button { min-height: 48px; min-width: 92px !important; font-weight: 750 !important; }
.caption-button { min-height: 43px; font-weight: 700 !important; }
.status-line { color: #64748b; font-size: .82rem; min-height: 1.25rem; }
.model-card { padding: .7rem .8rem; border-radius: 14px; color: #315f56; background: #eefbf7;
  border: 1px solid #ccefe5; font-size: .8rem; line-height: 1.5; }
.voice-panel { padding: .35rem !important; border-radius: 16px !important; background: #f7f8fc !important; }
.confidence-box { padding: .55rem .7rem; border-radius: 12px; background: #f8fafc;
  border: 1px solid rgba(100,116,139,.15); font-size: .84rem; }
.footer-note { text-align: center; color: #64748b; padding: .65rem 0 0; font-size: .8rem; }
@media (max-width: 760px) { .online { display:none; } .topbar { border-radius: 14px; } }
"""


def app_theme(gr):
    return gr.themes.Soft(
        primary_hue="indigo", secondary_hue="emerald", neutral_hue="slate", radius_size="lg"
    )


def create_app(captioner=None, vqa=None, speech=None, assessor=None, aligner=None):
    import gradio as gr

    captioner = captioner or BlipCaptioner()
    vqa = vqa or BlipVQA()
    speech = speech or SpeechService()
    assessor = assessor or ConfidenceAssessor(load_policy())
    aligner = aligner or ClipAligner()

    welcome = [{"role": "assistant", "content": "Xin chào! Hãy tải một ảnh lên, sau đó hỏi tôi bất kỳ điều gì về ảnh đó."}]

    def config(beams, tokens, temp, sample):
        return GenerationConfig(
            num_beams=int(beams),
            max_new_tokens=int(tokens),
            temperature=float(temp) if sample else 1.0,
            do_sample=bool(sample),
        )

    def evidence_config(generation, check_confidence):
        if check_confidence and (generation.num_beams < 2 or generation.do_sample):
            return replace(generation, num_beams=3, do_sample=False, temperature=1.0)
        return generation

    def infer_with_evidence(adapter, evidence_method, text_method, *args):
        if hasattr(adapter, evidence_method):
            return getattr(adapter, evidence_method)(*args)
        return GenerationEvidence(text=getattr(adapter, text_method)(*args))

    def confidence_for(result, image, use_clip):
        clip_similarity = None
        warning = None
        if use_clip:
            try:
                clip_similarity = aligner.similarity(image, result.text)
            except Exception as exc:
                warning = f"CLIP không khả dụng: {exc}"
        evidence = ConfidenceEvidence(
            mean_logprob=result.mean_logprob,
            beam_margin=result.beam_margin,
            primary_text=result.text,
            alternative_text=result.alternative_text,
            clip_similarity=clip_similarity,
        )
        return assessor.assess(evidence), warning

    def spoken(text, enabled, language, tts_backend):
        if not enabled:
            return None, None
        try:
            return str(speech.synthesize(text, language if language in {"vi", "en"} else "vi", tts_backend)), None
        except Exception as exc:
            return None, f"Không thể đọc phản hồi: {exc}"

    def caption(image, chat, beams, tokens, temp, sample, check_confidence, use_clip, auto_speak, language, tts_backend):
        chat = list(chat or welcome)
        if image is None:
            chat.append({"role": "assistant", "content": "Bạn cần tải ảnh lên trước khi tạo mô tả."})
            return chat, "Chưa có ảnh", "", None, "**Chưa đánh giá**"
        started = time.perf_counter()
        try:
            generation = evidence_config(config(beams, tokens, temp, sample), check_confidence)
            result = infer_with_evidence(captioner, "caption_with_evidence", "caption", image, generation)
            latency = (time.perf_counter() - started) * 1000
            decision, confidence_warning = confidence_for(result, image, use_clip) if check_confidence else (None, None)
            display = result.text
            if decision and decision.should_abstain:
                display = "Tôi chưa thể tạo mô tả đáng tin cậy từ ảnh này."
            elif decision and decision.level == "review":
                display += "\n\n_⚠ Nội dung này cần được đối chiếu với ảnh._"
            chat.append({"role": "assistant", "content": f"**Mô tả ảnh:** {display}"})
            audio_path, voice_error = spoken(display, auto_speak, language, tts_backend)
            status = f"Đã tạo caption · {latency:.2f} ms"
            if voice_error:
                status += f" · {voice_error}"
            if confidence_warning:
                status += f" · {confidence_warning}"
            confidence_text = render_confidence(decision) if decision else "**Chưa đánh giá** · Kiểm tra bằng chứng đang tắt."
            return chat, status, display, audio_path, confidence_text
        except Exception as exc:
            chat.append({"role": "assistant", "content": f"Không thể phân tích ảnh: {exc}"})
            return chat, "Tạo caption thất bại", "", None, "**Chưa đánh giá**"

    def answer(image, question, chat, records, beams, tokens, temp, sample, check_confidence, use_clip, auto_speak, language, tts_backend):
        chat = list(chat or welcome)
        records = list(records or [])
        if not question or not question.strip():
            return chat, records, "", "Hãy nhập câu hỏi", "", None, "**Chưa đánh giá**"
        if image is None:
            chat.append({"role": "user", "content": question.strip()})
            chat.append({"role": "assistant", "content": "Bạn cần tải ảnh lên trước để tôi có thể trả lời."})
            return chat, records, "", "Chưa có ảnh", "", None, "**Chưa đánh giá**"
        prompt = question.strip()
        chat.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        try:
            generation = evidence_config(config(beams, tokens, temp, sample), check_confidence)
            result = infer_with_evidence(vqa, "answer_with_evidence", "answer", image, prompt, generation)
        except Exception as exc:
            chat.append({"role": "assistant", "content": f"Không thể trả lời: {exc}"})
            return chat, records, "", "Xử lý thất bại", "", None, "**Chưa đánh giá**"
        session = ConversationSession()
        for item in records:
            session.add(item["question"], item["answer"], item["latency_ms"], item["settings"])
        latency = (time.perf_counter() - started) * 1000
        decision, confidence_warning = confidence_for(result, image, use_clip) if check_confidence else (None, None)
        display = result.text
        if decision and decision.should_abstain:
            display = "Tôi chưa thể xác định điều này từ ảnh."
        elif decision and decision.level == "review":
            display += "\n\n_⚠ Câu trả lời này cần được kiểm tra lại trên ảnh._"
        updated = session.add(prompt, result.text, latency, generation.__dict__)
        chat.append({"role": "assistant", "content": display})
        audio_path, voice_error = spoken(display.split("\n\n_")[0], auto_speak, language, tts_backend)
        status = f"Đã trả lời · {latency:.2f} ms"
        if voice_error:
            status += f" · {voice_error}"
        if confidence_warning:
            status += f" · {confidence_warning}"
        confidence_text = render_confidence(decision) if decision else "**Chưa đánh giá** · Kiểm tra bằng chứng đang tắt."
        return chat, updated, "", status, display, audio_path, confidence_text

    def transcribe(audio_path, language, speech_model):
        try:
            result = speech.transcribe(audio_path, language, speech_model)
            selected_language = result.language if result.language in {"vi", "en"} else language
            return (
                result.text,
                selected_language,
                f"Đã nhận dạng đủ audio {result.audio_seconds:.1f}s · {result.latency_ms:.0f} ms · {result.language} · Whisper {result.model_size}",
            )
        except Exception as exc:
            return "", language, f"STT thất bại: {exc}"

    def speak_response(text, language, tts_backend):
        if not text:
            return None, "Chưa có phản hồi để đọc"
        try:
            return str(
                speech.synthesize(
                    text.split("\n\n_")[0],
                    language if language in {"vi", "en"} else "vi",
                    tts_backend,
                )
            ), "Đã tạo âm thanh phản hồi"
        except Exception as exc:
            return None, f"TTS thất bại: {exc}"

    def reset_conversation():
        speech.cleanup()
        return welcome, [], "", "Cuộc trò chuyện mới", "", None, "**Chưa đánh giá**"

    def reset_for_image(_):
        speech.cleanup()
        return welcome, [], "Đã cập nhật ảnh · Bạn có thể bắt đầu hỏi", "", None, "**Chưa đánh giá**"

    with gr.Blocks(title="BLIP Vision Assistant") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML("""<header class="topbar"><div class="brand"><div class="brand-mark">✦</div>
            <div><h1>BLIP Vision Assistant</h1><p>Trợ lý hội thoại hiểu nội dung hình ảnh</p></div></div>
            <span class="online">● MODEL READY</span></header>""")

            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=290, elem_classes="sidebar"):
                    gr.Markdown("### Ngữ cảnh hình ảnh\nTải ảnh để bắt đầu cuộc hội thoại.", elem_classes="sidebar-title")
                    image = gr.Image(type="pil", label="Ảnh đang trò chuyện", height=300)
                    caption_btn = gr.Button("✦ Tạo mô tả ảnh", variant="secondary", elem_classes="caption-button")
                    new_chat = gr.Button("＋ Cuộc trò chuyện mới", variant="secondary")
                    with gr.Accordion("Giọng nói", open=True, elem_classes="voice-panel"):
                        language = gr.Dropdown(
                            choices=[("Tự động nhận diện", "auto"), ("Tiếng Việt", "vi"), ("English", "en")],
                            value="vi",
                            label="Ngôn ngữ",
                        )
                        speech_model = gr.Dropdown(
                            choices=[("Whisper Large-v3 — độ chính xác cao", "large-v3")],
                            value="large-v3",
                            label="Model nhận dạng",
                        )
                        tts_backend = gr.Dropdown(
                            choices=[
                                ("Edge TTS — online", "edge"),
                                ("Piper — offline, cần cấu hình", "piper"),
                            ],
                            value="edge",
                            label="Model đọc phản hồi",
                        )
                        microphone = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                            label="Ghi âm câu hỏi",
                            buttons=[],
                        )
                        transcribe_btn = gr.Button("Chuyển giọng nói thành văn bản", variant="secondary")
                        auto_speak = gr.Checkbox(label="Tự động đọc phản hồi", value=False)
                    with gr.Accordion("Thiết lập phản hồi", open=False):
                        gr.Markdown("`num_beams = 1`: Greedy · `num_beams > 1`: Beam Search")
                        beams = gr.Slider(1, 5, value=1, step=1, label="Số beam")
                        tokens = gr.Slider(5, 60, value=30, step=1, label="Token tối đa")
                        temp = gr.Slider(0.1, 2, value=1, step=0.1, label="Temperature")
                        sample = gr.Checkbox(label="Bật sampling")
                        check_confidence = gr.Checkbox(label="Kiểm tra bằng chứng và abstention", value=True)
                        use_clip = gr.Checkbox(label="Thêm tín hiệu CLIP (tải model ở lần đầu)", value=False)
                    gr.HTML('<div class="model-card"><b>BLIP Multimodal</b><br>Captioning Base · VQA Base</div>')

                with gr.Column(scale=8, min_width=420, elem_classes="chat-panel"):
                    gr.HTML('<div class="chat-heading"><h2>Hội thoại với hình ảnh</h2><p>Hỏi về đồ vật, màu sắc, vị trí hoặc nội dung trong ảnh.</p></div>')
                    chatbot = gr.Chatbot(
                        value=welcome,
                        label="",
                        show_label=False,
                        layout="bubble",
                        height=520,
                        placeholder="Tải ảnh và bắt đầu đặt câu hỏi",
                        buttons=["copy"],
                        elem_classes="chat-window",
                    )
                    records = gr.State([])
                    last_response = gr.State("")
                    with gr.Row(elem_classes="composer"):
                        question = gr.Textbox(
                            label="",
                            show_label=False,
                            placeholder="Nhập câu hỏi về hình ảnh...",
                            lines=1,
                            max_lines=4,
                            scale=8,
                        )
                        send = gr.Button("Gửi", variant="primary", scale=1, elem_classes="send-button")
                    with gr.Row():
                        speak_btn = gr.Button("🔊 Đọc phản hồi gần nhất", variant="secondary")
                        response_audio = gr.Audio(
                            label="Âm thanh phản hồi",
                            autoplay=True,
                            interactive=False,
                            buttons=[],
                        )
                    confidence_status = gr.Markdown(
                        "**Chưa đánh giá** · Gửi câu hỏi hoặc tạo caption để kiểm tra.",
                        elem_classes="confidence-box",
                    )
                    status = gr.Markdown("Sẵn sàng", elem_classes="status-line")

            gr.HTML('<div class="footer-note">Kết quả AI có thể chưa chính xác. Hãy đối chiếu với nội dung ảnh khi sử dụng.</div>')

        caption_inputs = [image, chatbot, beams, tokens, temp, sample, check_confidence, use_clip, auto_speak, language, tts_backend]
        answer_inputs = [image, question, chatbot, records, beams, tokens, temp, sample, check_confidence, use_clip, auto_speak, language, tts_backend]
        answer_outputs = [chatbot, records, question, status, last_response, response_audio, confidence_status]
        caption_btn.click(caption, caption_inputs, [chatbot, status, last_response, response_audio, confidence_status])
        send.click(answer, answer_inputs, answer_outputs)
        question.submit(answer, answer_inputs, answer_outputs)
        transcribe_btn.click(transcribe, [microphone, language, speech_model], [question, language, status])
        speak_btn.click(speak_response, [last_response, language, tts_backend], [response_audio, status])
        image.change(reset_for_image, image, [chatbot, records, status, last_response, response_audio, confidence_status])
        new_chat.click(
            reset_conversation,
            outputs=[chatbot, records, question, status, last_response, response_audio, confidence_status],
        )

    return demo


if __name__ == "__main__":
    import gradio as gr

    create_app().launch(theme=app_theme(gr), css=APP_CSS)
