import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram, butter, filtfilt
import streamlit_authenticator as stauth
from st_gsheets_connection import GSheetsConnection
import pandas as pd
import librosa
import io

# ============================================================
# 1. إعداد الصفحة
# ============================================================
st.set_page_config(page_title="Radar Signal Cloud Intelligence", layout="wide")

st.markdown("""
    <style>
    .stAppDeployButton, #StyledgithubIcon, [data-testid="bundle_github_cursor_detector"] {
        display: none !important;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. Google Sheets Connection
# ============================================================
url_sheet = "https://docs.google.com/spreadsheets/d/13kcl0WS0LE1rXWm4aanpby8wO5542JaR76038ofa1-E/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    try:
        df = conn.read(spreadsheet=url_sheet, ttl=0)
        creds = {"usernames": {}}
        if df is not None and not df.empty:
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                u_name = str(row['Username']).strip()
                if u_name and u_name != 'nan':
                    creds["usernames"][u_name] = {
                        "name": str(row['Name']),
                        "password": str(row['Password']),
                        "email": str(row.get('Email', ''))
                    }
        return creds
    except Exception:
        return {"usernames": {}}

credentials = get_all_users()

authenticator = stauth.Authenticate(
    credentials,
    "radar_intelligence_cookie",
    "auth_signature_key_2026",
    cookie_expiry_days=30
)

# ============================================================
# 3. Login / Register
# ============================================================
if not st.session_state.get('authentication_status'):
    tab1, tab2 = st.tabs(["Login", "Register New Engineer"])

    with tab2:
        try:
            registration_result = authenticator.register_user(location='main')
            if registration_result:
                usernames = list(credentials["usernames"].keys())
                new_username = usernames[-1]
                user_info = credentials["usernames"][new_username]
                if st.session_state.get('last_registered') != new_username:
                    with st.spinner('Cloud Syncing...'):
                        new_entry = pd.DataFrame([{
                            'Name': user_info.get('name', 'Eng. User'),
                            'Last name': 'Engineer',
                            'Email': user_info.get('email', 'N/A'),
                            'Username': new_username,
                            'Password': user_info.get('password', ''),
                            'Captcha': 'Verified',
                            'Password hint': 'Radar Project'
                        }])
                        existing_df = conn.read(spreadsheet=url_sheet, ttl=0)
                        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                        conn.update(spreadsheet=url_sheet, data=updated_df)
                        st.session_state['last_registered'] = new_username
                        st.success('✅ Registration Successful! Please switch to Login tab.')
                        st.balloons()
        except Exception as e:
            st.error(f"Registration Error: {e}")

    with tab1:
        authenticator.login(location='main')
        if st.session_state.get("authentication_status"):
            st.rerun()
        elif st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')

# ============================================================
# 4. الصفحة الرئيسية (بعد الـ Login)
# ============================================================
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')

    with st.sidebar:
        st.success(f"Welcome, Eng. {st.session_state.get('name', 'User')}")
        st.markdown("---")
        st.subheader("🌐 System Infrastructure")
        st.info("**Environment:** Docker Container")
        st.info("**Database:** Google Cloud Real-time")

    st.title("📡 Radar Signal Intelligence System")
    st.markdown("---")

    # ============================================================
    # 5. الفلاتر (نفس اللي في Train_CNN)
    # ============================================================
    def low_pass_filter(signal, cutoff=300, fs=5000, order=5):
        nyq = fs / 2
        b, a = butter(order, cutoff / nyq, btype='low')
        return filtfilt(b, a, signal)

    def high_pass_filter(signal, cutoff=20, fs=5000, order=5):
        nyq = fs / 2
        b, a = butter(order, cutoff / nyq, btype='high')
        return filtfilt(b, a, signal)

    def band_pass_filter(signal, low=40, high=200, fs=5000, order=5):
        nyq = fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, signal)

    def apply_best_filter(signal):
        """
        بيجرب يطبق الفلتر الأنسب تلقائياً على الإشارة القادمة من الـ audio
        """
        signal = high_pass_filter(signal)  # شيل الـ DC offset دايماً
        return low_pass_filter(signal, cutoff=300)

    def get_spec(signal, fs=5000):
        _, _, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    ALL_LABELS = ["AM Signal", "FM Signal", "FSK", "4-QAM", "16-QAM"]

    # ============================================================
    # 6. الـ Tabs الرئيسية
    # ============================================================
    main_tab1, main_tab2 = st.tabs(["🎛️ Generate & Classify", "📂 Upload Audio File"])

    # ==================== TAB 1: Generate ====================
    with main_tab1:
        col_ctrl, col_res = st.columns([1, 2])

        with col_ctrl:
            st.subheader("Signal Generation")
            signal_option = st.selectbox(
                "Select Modulation:",
                ["AM Signal", "FM Signal", "FSK", "4-QAM", "16-QAM"]
            )
            gen_btn = st.button("Generate & Classify 🚀")

        if gen_btn:
            model = load_my_model()
            t = np.linspace(0, 1, 5000, endpoint=False)

            if signal_option == "AM Signal":
                signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
            elif signal_option == "FM Signal":
                signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / 5000))
            elif signal_option == "FSK":
                f1, f2 = 50, 150
                data = np.repeat(np.random.randint(0, 2, 10), 500)
                freqs = np.where(data == 0, f1, f2)
                signal = np.sin(2 * np.pi * freqs * t)
            elif signal_option == "4-QAM":
                phases = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
                signal = np.sin(2 * np.pi * 100 * t + np.random.choice(phases))
            elif signal_option == "16-QAM":
                points = [-3, -1, 1, 3]
                I, Q = np.random.choice(points), np.random.choice(points)
                signal = (I * np.cos(2 * np.pi * 100 * t) - Q * np.sin(2 * np.pi * 100 * t))
                signal = signal / np.max(np.abs(signal))

            with col_res:
                st.subheader("1. Time Domain")
                fig1, ax1 = plt.subplots(figsize=(10, 3))
                ax1.plot(t[:500], signal[:500], color='dodgerblue')
                st.pyplot(fig1)
                plt.close()

                if "QAM" in signal_option:
                    st.subheader("2. Constellation Diagram")
                    fig_const, ax_const = plt.subplots(figsize=(4, 4))
                    if signal_option == "4-QAM":
                        pts = [1+1j, 1-1j, -1+1j, -1-1j]
                    else:
                        pts = [complex(i, q) for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]
                    ax_const.scatter([p.real for p in pts], [p.imag for p in pts], color='red', marker='x')
                    ax_const.grid(True)
                    ax_const.axhline(0, color='black')
                    ax_const.axvline(0, color='black')
                    st.pyplot(fig_const)
                    plt.close()

                with st.spinner('Analyzing...'):
                    spec = get_spec(signal)
                    prediction = model.predict(spec.reshape(1, 129, 38, 1))
                    res_label = ALL_LABELS[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100

                    st.subheader("3. Spectrogram")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                    st.pyplot(fig2)
                    plt.close()

                    st.subheader("4. Intelligence Results")
                    c1, c2 = st.columns(2)
                    c1.metric("Detected Signal Type", res_label)
                    c2.metric("Model Confidence", f"{confidence:.2f}%")

                    # شريط الـ confidence لكل الأنواع
                    st.subheader("5. Confidence per Class")
                    fig3, ax3 = plt.subplots(figsize=(8, 3))
                    colors = ['green' if i == np.argmax(prediction) else 'steelblue'
                              for i in range(5)]
                    ax3.barh(ALL_LABELS, prediction[0] * 100, color=colors)
                    ax3.set_xlabel('Confidence %')
                    ax3.set_xlim(0, 100)
                    st.pyplot(fig3)
                    plt.close()

    # ==================== TAB 2: Upload Audio ====================
    with main_tab2:
        st.subheader("📂 Upload Signal Audio File")
        st.info("ارفع ملف audio بتاع الإشارة (wav, mp3, flac, ogg, ...) وهيتم تصنيفه تلقائياً")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg", "aiff", "au"]
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            with st.spinner('Loading & Processing Audio... 🔄'):
                try:
                    # قراءة الـ audio بأي format
                    audio_bytes = uploaded_file.read()
                    audio_array, sr = librosa.load(
                        io.BytesIO(audio_bytes),
                        sr=5000,        # resample لـ 5000 Hz زي التدريب
                        mono=True       # تحويل لـ mono
                    )

                    # لو الإشارة أطول من 5000 sample، خد أول 5000
                    # لو أقصر، كمّلها بأصفار
                    if len(audio_array) >= 5000:
                        audio_array = audio_array[:5000]
                    else:
                        audio_array = np.pad(audio_array, (0, 5000 - len(audio_array)))

                    # تطبيق الفلتر
                    filtered_signal = apply_best_filter(audio_array)

                    # عرض الإشارة
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.subheader("1. Original Signal")
                        t_axis = np.linspace(0, 1, 5000)
                        fig_raw, ax_raw = plt.subplots(figsize=(6, 3))
                        ax_raw.plot(t_axis[:500], audio_array[:500], color='red', alpha=0.7)
                        ax_raw.set_title("Before Filter")
                        st.pyplot(fig_raw)
                        plt.close()

                    with col_b:
                        st.subheader("2. Filtered Signal")
                        fig_filt, ax_filt = plt.subplots(figsize=(6, 3))
                        ax_filt.plot(t_axis[:500], filtered_signal[:500], color='dodgerblue')
                        ax_filt.set_title("After Filter")
                        st.pyplot(fig_filt)
                        plt.close()

                    # Spectrogram
                    spec = get_spec(filtered_signal)

                    st.subheader("3. Spectrogram")
                    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
                    ax_spec.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                    ax_spec.set_title("Signal Spectrogram")
                    st.pyplot(fig_spec)
                    plt.close()

                    # التصنيف
                    with st.spinner('Classifying... 🤖'):
                        model = load_my_model()
                        prediction = model.predict(spec.reshape(1, 129, 38, 1))
                        res_label  = ALL_LABELS[np.argmax(prediction)]
                        confidence = np.max(prediction) * 100

                    # النتيجة
                    st.markdown("---")
                    st.subheader("4. Classification Result")

                    # لون حسب الـ confidence
                    if confidence >= 80:
                        result_color = "🟢"
                    elif confidence >= 50:
                        result_color = "🟡"
                    else:
                        result_color = "🔴"

                    c1, c2, c3 = st.columns(3)
                    c1.metric("📡 Detected Signal", res_label)
                    c2.metric("🎯 Confidence", f"{confidence:.2f}%")
                    c3.metric("📁 File", uploaded_file.name)

                    st.info(f"{result_color} **{res_label}** detected with **{confidence:.2f}%** confidence")

                    # شريط الـ confidence لكل الأنواع
                    st.subheader("5. Confidence per Class")
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
                    colors = ['green' if i == np.argmax(prediction) else 'steelblue'
                              for i in range(5)]
                    ax_bar.barh(ALL_LABELS, prediction[0] * 100, color=colors)
                    ax_bar.set_xlabel('Confidence %')
                    ax_bar.set_xlim(0, 100)
                    st.pyplot(fig_bar)
                    plt.close()

                    # معلومات الملف
                    with st.expander("📊 Audio File Details"):
                        st.write(f"**Sample Rate:** {sr} Hz (resampled to 5000 Hz)")
                        st.write(f"**Duration:** {len(audio_array)/5000:.2f} seconds")
                        st.write(f"**Samples:** {len(audio_array)}")
                        st.write(f"**File Size:** {len(audio_bytes)/1024:.1f} KB")

                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    st.warning("تأكد إن الملف audio صحيح وحاول تاني")
