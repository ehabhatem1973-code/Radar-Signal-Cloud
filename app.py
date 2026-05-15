import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram, butter, filtfilt
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import librosa
import io

# ============================================================
# 1. إعداد الصفحة
# ============================================================
st.set_page_config(page_title="Radar Signal Cloud Intelligence", layout="wide")

st.markdown("""
    <style>
    .stAppDeployButton, #StyledgithubIcon,
    [data-testid="bundle_github_cursor_detector"] { display: none !important; }
    footer { visibility: hidden; }
    .param-card {
        background: #1e2a3a;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .param-label { font-size: 12px; color: #8ab4d4; margin-bottom: 2px; }
    .param-value { font-size: 20px; font-weight: 700; color: #00d4ff; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. الـ 12 إشارة + معلوماتها
# ============================================================
ALL_LABELS = [
    "AM", "FM", "PM",
    "SSB", "DSB",
    "FSK", "ASK", "BPSK",
    "QPSK", "4-QAM", "16-QAM", "64-QAM"
]

SIGNAL_INFO = {
    "AM"    : {"type": "Analog",  "full": "Amplitude Modulation",
               "desc": "Carrier amplitude varies with the message. Used in AM radio broadcasting."},
    "FM"    : {"type": "Analog",  "full": "Frequency Modulation",
               "desc": "Carrier frequency varies with message. Better noise immunity. Used in FM radio."},
    "PM"    : {"type": "Analog",  "full": "Phase Modulation",
               "desc": "Carrier phase varies with message. Foundation of digital PSK."},
    "SSB"   : {"type": "Analog",  "full": "Single Side Band",
               "desc": "One sideband transmitted — doubles spectral efficiency over DSB."},
    "DSB"   : {"type": "Analog",  "full": "Double Side Band",
               "desc": "Both sidebands, carrier suppressed. Used in AM stereo."},
    "FSK"   : {"type": "Digital", "full": "Frequency Shift Keying",
               "desc": "Data encoded by switching frequencies. Robust to amplitude noise."},
    "ASK"   : {"type": "Digital", "full": "Amplitude Shift Keying",
               "desc": "Data encoded in amplitude changes. Simple but noise-sensitive."},
    "BPSK"  : {"type": "Digital", "full": "Binary Phase Shift Keying",
               "desc": "Two phase states (0°/180°). Very robust — used in GPS & satellite."},
    "QPSK"  : {"type": "Digital", "full": "Quadrature Phase Shift Keying",
               "desc": "4 phase states, 2 bits/symbol. Used in DVB, CDMA, LTE uplink."},
    "4-QAM" : {"type": "Digital", "full": "4-Quadrature Amplitude Modulation",
               "desc": "Amplitude & phase combined (4 points). Equivalent to QPSK."},
    "16-QAM": {"type": "Digital", "full": "16-Quadrature Amplitude Modulation",
               "desc": "16 constellation points, 4 bits/symbol. Used in Wi-Fi & LTE."},
    "64-QAM": {"type": "Digital", "full": "64-Quadrature Amplitude Modulation",
               "desc": "64 points, 6 bits/symbol. High throughput — used in cable TV & 5G."},
}

FS = 5000

# ============================================================
# 3. Google Sheets Connection
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
                        "name"    : str(row['Name']),
                        "password": str(row['Password']),
                        "email"   : str(row.get('Email', ''))
                    }
        return creds
    except Exception:
        return {"usernames": {}}

credentials   = get_all_users()
authenticator = stauth.Authenticate(
    credentials,
    "radar_intelligence_cookie",
    "auth_signature_key_2026",
    cookie_expiry_days=30
)

# ============================================================
# 4. Login / Register
# ============================================================
if not st.session_state.get('authentication_status'):
    tab1, tab2 = st.tabs(["Login", "Register New Engineer"])

    with tab2:
        try:
            registration_result = authenticator.register_user(location='main')
            if registration_result:
                usernames    = list(credentials["usernames"].keys())
                new_username = usernames[-1]
                user_info    = credentials["usernames"][new_username]
                if st.session_state.get('last_registered') != new_username:
                    with st.spinner('Cloud Syncing...'):
                        new_entry   = pd.DataFrame([{
                            'Name'        : user_info.get('name', 'Eng. User'),
                            'Last name'   : 'Engineer',
                            'Email'       : user_info.get('email', 'N/A'),
                            'Username'    : new_username,
                            'Password'    : user_info.get('password', ''),
                            'Captcha'     : 'Verified',
                            'Password hint': 'Radar Project'
                        }])
                        existing_df = conn.read(spreadsheet=url_sheet, ttl=0)
                        updated_df  = pd.concat([existing_df, new_entry], ignore_index=True)
                        conn.update(spreadsheet=url_sheet, data=updated_df)
                        st.session_state['last_registered'] = new_username
                        st.success('Registration Successful! Switch to Login tab.')
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
# 5. الصفحة الرئيسية
# ============================================================
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')

    with st.sidebar:
        st.success(f"Welcome, Eng. {st.session_state.get('name', 'User')}")
        st.markdown("---")
        st.subheader("🌐 System Infrastructure")
        st.info("**Environment:** Docker Container")
        st.info("**Database:** Google Cloud Real-time")
        st.markdown("---")
        st.subheader("📡 Supported Signals")
        st.markdown("**Analog:** AM · FM · PM · SSB · DSB")
        st.markdown("**Digital:** FSK · ASK · BPSK · QPSK · 4-QAM · 16-QAM · 64-QAM")

    st.title("📡 Radar Signal Intelligence System")
    st.markdown("---")

    # ============================================================
    # 6. الفلاتر
    # ============================================================
    def low_pass_filter(signal, cutoff=300, fs=FS, order=5):
        b, a = butter(order, cutoff / (fs/2), btype='low')
        return filtfilt(b, a, signal)

    def high_pass_filter(signal, cutoff=20, fs=FS, order=5):
        b, a = butter(order, cutoff / (fs/2), btype='high')
        return filtfilt(b, a, signal)

    def band_pass_filter(signal, low=40, high=200, fs=FS, order=5):
        nyq = fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, signal)

    def apply_signal_filter(signal, signal_type=""):
        signal = high_pass_filter(signal)
        if signal_type in ["FM", "FSK", "PM"]:
            return band_pass_filter(signal)
        return low_pass_filter(signal)

    def get_spec(signal, fs=FS):
        _, _, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
        Sxx_log   = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)

    # ============================================================
    # 7. استخراج الـ Parameters
    # ============================================================
    def estimate_parameters(signal, fs=FS):
        N        = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs    = np.fft.rfftfreq(N, d=1/fs)

        carrier_freq = float(freqs[np.argmax(fft_vals)])

        peak_power = np.max(fft_vals**2)
        above      = fft_vals**2 >= (peak_power / 2)
        bw = float(freqs[above][-1] - freqs[above][0]) if np.any(above) else 0.0

        rms    = np.sqrt(np.mean(signal**2))
        active = (np.abs(signal) > rms).astype(int)
        diff_  = np.diff(active)
        starts = np.where(diff_ == 1)[0]
        ends   = np.where(diff_ == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            if ends[0] < starts[0]:
                ends = ends[1:]
            n = min(len(starts), len(ends))
            pw = float(np.mean((ends[:n] - starts[:n]) / fs * 1e6))
        else:
            pw = 0.0

        pri = float(np.mean(np.diff(starts)) / fs * 1e6) if len(starts) > 1 else 0.0

        return {
            "Carrier Frequency": f"{carrier_freq:.1f} Hz",
            "Bandwidth"        : f"{bw:.1f} Hz",
            "Pulse Width"      : f"{pw:.1f} µs",
            "PRI"              : f"{pri:.1f} µs",
        }

    # ============================================================
    # 8. توليد الإشارات - TAB 1
    # ============================================================
    def generate_signal(name, fs=FS):
        t = np.linspace(0, 1, fs, endpoint=False)
        if name == "AM":
            return (1 + 0.5*np.sin(2*np.pi*5*t)) * np.sin(2*np.pi*100*t)
        elif name == "FM":
            return np.sin(2*np.pi*(100*t + 20*np.cumsum(np.sin(2*np.pi*5*t))/fs))
        elif name == "PM":
            return np.sin(2*np.pi*100*t + np.pi*np.sin(2*np.pi*5*t))
        elif name == "SSB":
            m = np.sin(2*np.pi*5*t)
            return m*np.cos(2*np.pi*100*t) - np.sqrt(1-m**2+1e-8)*np.sin(2*np.pi*100*t)
        elif name == "DSB":
            return np.sin(2*np.pi*5*t) * np.cos(2*np.pi*100*t)
        elif name == "FSK":
            data = np.repeat(np.random.randint(0,2,10), 500)
            return np.sin(2*np.pi*np.where(data==0,50,150)*t)
        elif name == "ASK":
            data = np.repeat(np.random.randint(0,2,10), 500)
            return np.where(data==0, 0.1, 1.0) * np.sin(2*np.pi*100*t)
        elif name == "BPSK":
            data = np.repeat(np.random.randint(0,2,10), 500)
            return np.sin(2*np.pi*100*t + np.where(data==0,0,np.pi))
        elif name == "QPSK":
            return np.sin(2*np.pi*100*t + np.random.choice([0,np.pi/2,np.pi,3*np.pi/2]))
        elif name == "4-QAM":
            return np.sin(2*np.pi*100*t + np.random.choice([np.pi/4,3*np.pi/4,5*np.pi/4,7*np.pi/4]))
        elif name == "16-QAM":
            pts = [-3,-1,1,3]; I,Q = np.random.choice(pts), np.random.choice(pts)
            sig = I*np.cos(2*np.pi*100*t) - Q*np.sin(2*np.pi*100*t)
            return sig/(np.max(np.abs(sig))+1e-8)
        elif name == "64-QAM":
            pts = [-7,-5,-3,-1,1,3,5,7]; I,Q = np.random.choice(pts), np.random.choice(pts)
            sig = I*np.cos(2*np.pi*100*t) - Q*np.sin(2*np.pi*100*t)
            return sig/(np.max(np.abs(sig))+1e-8)
        return np.zeros(fs)

    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    # ============================================================
    # 9. دالة عرض النتائج المشتركة
    # ============================================================
    def show_results(raw_signal, filtered_signal, model):
        spec = get_spec(filtered_signal)

        # Spectrogram
        st.subheader("📊 Spectrogram")
        fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
        img = ax_sp.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax_sp.set_title("Signal Spectrogram")
        ax_sp.set_xlabel("Time"); ax_sp.set_ylabel("Frequency")
        plt.colorbar(img, ax=ax_sp, label='Normalized Power (dB)')
        st.pyplot(fig_sp); plt.close()

        # التصنيف
        with st.spinner('🤖 Classifying...'):
            prediction = model.predict(spec.reshape(1, 129, 38, 1), verbose=0)
            idx        = int(np.argmax(prediction))
            res_label  = ALL_LABELS[idx]
            confidence = float(np.max(prediction)) * 100

        st.markdown("---")
        st.subheader("🎯 Classification Result")

        badge = "🟢" if confidence >= 80 else ("🟡" if confidence >= 50 else "🔴")
        c1, c2, c3 = st.columns(3)
        c1.metric("📡 Signal Type", res_label)
        c2.metric("🎯 Confidence",  f"{confidence:.1f}%")
        c3.metric("📶 Category",    SIGNAL_INFO[res_label]["type"])

        st.info(f"{badge} **{SIGNAL_INFO[res_label]['full']}** detected with **{confidence:.1f}%** confidence")
        st.caption(SIGNAL_INFO[res_label]["desc"])

        # Parameters
        st.subheader("📐 Signal Parameters")
        params = estimate_parameters(filtered_signal)
        p_cols = st.columns(4)
        icons  = ["📻", "📡", "⏱️", "🔄"]
        for col, (k, v), icon in zip(p_cols, params.items(), icons):
            col.markdown(
                f'<div class="param-card">'
                f'<div class="param-label">{icon} {k}</div>'
                f'<div class="param-value">{v}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Confidence Bar
        st.subheader("📊 Confidence per Class")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        colors = ['#00d4ff' if i == idx else '#2d5a7a' for i in range(len(ALL_LABELS))]
        bars   = ax_bar.barh(ALL_LABELS, prediction[0] * 100, color=colors)
        ax_bar.set_xlabel('Confidence %'); ax_bar.set_xlim(0, 100)
        ax_bar.bar_label(bars, fmt='%.1f%%', padding=3)
        ax_bar.set_facecolor('#0d1117'); fig_bar.patch.set_facecolor('#0d1117')
        ax_bar.tick_params(colors='white'); ax_bar.xaxis.label.set_color('white')
        plt.tight_layout()
        st.pyplot(fig_bar); plt.close()

        # Constellation
        if res_label in ["BPSK", "QPSK", "4-QAM", "16-QAM", "64-QAM"]:
            st.subheader("🔵 Constellation Diagram")
            fig_c, ax_c = plt.subplots(figsize=(5, 5))
            if res_label == "BPSK":
                pts = [1+0j, -1+0j]
            elif res_label in ["QPSK", "4-QAM"]:
                pts = [1+1j, -1+1j, -1-1j, 1-1j]
            elif res_label == "16-QAM":
                pts = [complex(i,q) for i in [-3,-1,1,3] for q in [-3,-1,1,3]]
            else:
                pts = [complex(i,q) for i in [-7,-5,-3,-1,1,3,5,7] for q in [-7,-5,-3,-1,1,3,5,7]]
            ax_c.scatter([p.real for p in pts], [p.imag for p in pts],
                         color='cyan', marker='x', s=80)
            ax_c.grid(True, alpha=0.3)
            ax_c.axhline(0, color='white', lw=0.5); ax_c.axvline(0, color='white', lw=0.5)
            ax_c.set_facecolor('#0d1117'); fig_c.patch.set_facecolor('#0d1117')
            ax_c.tick_params(colors='white')
            ax_c.set_title(f"{res_label} Constellation", color='white')
            st.pyplot(fig_c); plt.close()

    # ============================================================
    # 10. الـ Tabs الرئيسية
    # ============================================================
    main_tab1, main_tab2 = st.tabs(["🎛️ Generate & Classify", "📂 Upload Audio File"])

    # ================== TAB 1 ==================
    with main_tab1:
        col_ctrl, col_res = st.columns([1, 2])

        with col_ctrl:
            st.subheader("Signal Generation")
            signal_option = st.selectbox("Select Modulation:", ALL_LABELS)
            noise_level   = st.slider("Noise Level (σ)", 0.0, 1.0, 0.1, 0.05)
            gen_btn       = st.button("Generate & Classify 🚀", use_container_width=True)

            if signal_option in SIGNAL_INFO:
                info = SIGNAL_INFO[signal_option]
                st.markdown(f"""
                **Type:** {info['type']}  
                **Full Name:** {info['full']}  
                _{info['desc']}_
                """)

        if gen_btn:
            model      = load_my_model()
            t          = np.linspace(0, 1, FS, endpoint=False)
            raw_signal = generate_signal(signal_option)
            noise      = np.random.normal(0, noise_level, FS)
            signal     = raw_signal + noise
            filtered   = apply_signal_filter(signal, signal_option)

            with col_res:
                st.subheader("📈 Time Domain")
                fig1, axes = plt.subplots(2, 1, figsize=(10, 5))
                axes[0].plot(t[:500], signal[:500],   color='tomato',     alpha=0.8)
                axes[0].set_title("Raw Signal");      axes[0].set_ylabel("Amplitude")
                axes[1].plot(t[:500], filtered[:500], color='dodgerblue')
                axes[1].set_title("Filtered Signal"); axes[1].set_ylabel("Amplitude")
                axes[1].set_xlabel("Time (s)")
                plt.tight_layout()
                st.pyplot(fig1); plt.close()

                show_results(signal, filtered, model)

    # ================== TAB 2 ==================
    with main_tab2:
        st.subheader("📂 Upload Signal Audio File")
        st.info("ارفع ملف audio (wav, mp3, flac, ogg) وهيتم تصنيفه وتحليله تلقائياً مع استخراج كل الـ Parameters")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg", "aiff", "au"]
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            with st.spinner('🔄 Processing Audio...'):
                try:
                    audio_bytes         = uploaded_file.read()
                    audio_array, sr     = librosa.load(io.BytesIO(audio_bytes), sr=FS, mono=True)

                    if len(audio_array) >= FS:
                        audio_array = audio_array[:FS]
                    else:
                        audio_array = np.pad(audio_array, (0, FS - len(audio_array)))

                    filtered_signal = apply_signal_filter(audio_array)

                    col_a, col_b = st.columns(2)
                    t_axis = np.linspace(0, 1, FS)

                    with col_a:
                        st.subheader("📈 Original Signal")
                        fig_r, ax_r = plt.subplots(figsize=(6, 3))
                        ax_r.plot(t_axis[:500], audio_array[:500], color='tomato', alpha=0.7)
                        ax_r.set_title("Before Filter")
                        st.pyplot(fig_r); plt.close()

                    with col_b:
                        st.subheader("📈 Filtered Signal")
                        fig_f, ax_f = plt.subplots(figsize=(6, 3))
                        ax_f.plot(t_axis[:500], filtered_signal[:500], color='dodgerblue')
                        ax_f.set_title("After Filter")
                        st.pyplot(fig_f); plt.close()

                    model = load_my_model()
                    show_results(audio_array, filtered_signal, model)

                    with st.expander("📊 Audio File Details"):
                        st.write(f"**Original Sample Rate:** {sr} Hz")
                        st.write(f"**Resampled To:** {FS} Hz")
                        st.write(f"**Duration:** {len(audio_array)/FS:.2f} sec")
                        st.write(f"**Samples:** {len(audio_array)}")
                        st.write(f"**File Size:** {len(audio_bytes)/1024:.1f} KB")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.warning("تأكد إن الملف audio صحيح وحاول تاني")
