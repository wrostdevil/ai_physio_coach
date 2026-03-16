import streamlit as st
import av
import time
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from physio_coach import process_frame, stats
from report_generator import generate_report

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Physio Coach",
    page_icon="🏋️",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown("""
# 🏋️ AI Physio Coach
### Real-time AI Exercise Detection
""")

# ---------- SESSION TIMER ----------
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

elapsed = int(time.time() - st.session_state.start_time)

mins = elapsed // 60
secs = elapsed % 60

session_time = f"{mins}:{secs:02d}"

st.sidebar.title("Session Info")
st.sidebar.metric("Session Time", session_time)

# ---------- LAYOUT ----------
col1, col2 = st.columns([3,1])

# ---------- VIDEO PROCESSOR ----------
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = process_frame(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- CAMERA ----------
with col1:

    st.subheader("Live Camera")

    webrtc_streamer(
        key="ai-fitness",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

# ---------- DASHBOARD ----------
with col2:

    st.subheader("Workout Stats")

    st.metric("🦵 Squats", stats["squat"])
    st.metric("💪 Curls", stats["curl"])
    st.metric("🔥 Pushups", stats["pushup"])

    st.divider()

    st.subheader("Workout Progress")

    chart_data = pd.DataFrame({
        "Exercise": ["Squats","Curls","Pushups"],
        "Reps": [stats["squat"],stats["curl"],stats["pushup"]]
    })

    st.bar_chart(chart_data.set_index("Exercise"))

    st.divider()

    if st.button("Reset Workout"):
        stats["squat"] = 0
        stats["curl"] = 0
        stats["pushup"] = 0

    st.divider()
# ---------- REPORT SECTION ----------
st.subheader("Workout Report")

# create storage
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# generate report
if st.button("Generate Workout Report"):
    st.session_state.pdf_bytes = generate_report(stats, session_time)
    st.success("Report generated!")

# download button
if st.session_state.pdf_bytes is not None:

    st.download_button(
        label="Download Workout Report",
        data=st.session_state.pdf_bytes,
        file_name="workout_report.pdf",
        mime="application/pdf"
    )