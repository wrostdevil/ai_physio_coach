import streamlit as st
import av
import time
import pandas as pd
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from physio_coach import process_frame, stats
from report_generator import generate_report


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Physio Coach",
    layout="wide",
    page_icon="🏋️"
)


# ---------- RESPONSIVE STYLE ----------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
    color:white;
}

[data-testid="stSidebar"]{
    background-color:#111;
}

h1,h2,h3,h4{
    color:#00ffd5;
}

/* Responsive camera */
video {
    width:100% !important;
    height:auto !important;
    border-radius:12px;
}

/* Dashboard cards */
.metric-card{
    background:#1e1e1e;
    padding:15px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.4);
}

/* Mobile layout adjustments */
@media (max-width: 768px){

    h1{
        font-size:26px;
    }

    .metric-card{
        padding:10px;
    }

    [data-testid="stSidebar"]{
        display:none;
    }

}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.title("🏋️ AI Physio Coach")
st.markdown("### Real-time AI Exercise Detection & Workout Analytics")


# ---------- TIMER ----------
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

elapsed = int(time.time() - st.session_state.start_time)

mins = elapsed // 60
secs = elapsed % 60

session_time = f"{mins}:{secs:02d}"


# ---------- SIDEBAR ----------
st.sidebar.header("Session Info")
st.sidebar.metric("Session Time", session_time)

st.sidebar.write("### Exercises Detected")
st.sidebar.write("• Squats")
st.sidebar.write("• Curls")
st.sidebar.write("• Pushups")


# ---------- RESPONSIVE LAYOUT ----------
# On mobile Streamlit automatically stacks columns
col1, col2 = st.columns([4,2])


# ---------- VIDEO PROCESSOR ----------
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = process_frame(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- CAMERA ----------
with col1:

    st.subheader("Live Workout Camera")

    webrtc_streamer(
    key="ai-fitness",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)


# ---------- DASHBOARD ----------
with col2:

    st.subheader("Workout Dashboard")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric("🦵 Squats",stats["squat"])

    with c2:
        st.metric("💪 Curls",stats["curl"])

    with c3:
        st.metric("🔥 Pushups",stats["pushup"])


    st.divider()


    # ---------- WORKOUT INTENSITY ----------
    total_reps = stats["squat"] + stats["curl"] + stats["pushup"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_reps,
        title={'text':"Workout Intensity"},
        gauge={'axis':{'range':[0,50]}}
    ))

    st.plotly_chart(fig,use_container_width=True)


    st.divider()


    # ---------- PROGRESS CHART ----------
    st.subheader("Workout Progress")

    chart_data = pd.DataFrame({
        "Exercise":["Squats","Curls","Pushups"],
        "Reps":[stats["squat"],stats["curl"],stats["pushup"]]
    })

    st.bar_chart(chart_data.set_index("Exercise"),use_container_width=True)


    # ---------- RESET ----------
    if st.button("Reset Workout"):

        stats["squat"]=0
        stats["curl"]=0
        stats["pushup"]=0


    st.divider()


    # ---------- TRAINER TIPS ----------
    st.subheader("AI Trainer Tips")

    tips=[
        "Keep your back straight during squats",
        "Control movement during curls",
        "Maintain body alignment during pushups"
    ]

    for tip in tips:
        st.info(tip)


    st.divider()


    # ---------- REPORT ----------
    st.subheader("Workout Report")

    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes=None

    if st.button("Generate Workout Report"):
        st.session_state.pdf_bytes=generate_report(stats,session_time)

    if st.session_state.pdf_bytes:

        st.download_button(
            "Download Workout Report",
            data=st.session_state.pdf_bytes,
            file_name="workout_report.pdf",
            mime="application/pdf"
        )