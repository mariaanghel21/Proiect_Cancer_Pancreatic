import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import io

st.set_page_config(
    page_title="PancreasAI Diagnostic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource 
def load_model():
    return YOLO('runs/detect/pancreas_final_result-2/weights/best.pt')

model = load_model()

st.sidebar.title("🏥 PancreasAI v1.0")
st.sidebar.markdown("---")

st.sidebar.subheader("📋 Date Pacient")
pacient_id = st.sidebar.text_input("ID Examinare CT", "CT-2026-001")
pacient_varsta = st.sidebar.number_input("Varsta", min_value=1, max_value=120, value=55)
pacient_sex = st.sidebar.selectbox("Sex", ["Masculin", "Feminin", "Nespecificat"])

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Setari Analiza")
conf_threshold = st.sidebar.slider("Sensibilitate Model (Confidence)", 0.1, 1.0, 0.45)

tab1, tab2, tab3 = st.tabs(["🔍 Diagnostic AI", "📊 Statistici Proiect", "📖 Ghid Utilizare"])

with tab1:
    st.title("Sistem Computer-Aided Diagnosis (CAD)")
    st.write("Instrument digital pentru asistenta in identificarea formatiunilor tumorale pancreatice.")
    
    uploaded_file = st.file_uploader("Incarcati scanarea CT (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.markdown("### Vizualizare Scanare")
            st.image(image, use_container_width=True, caption=f"Original: {uploaded_file.name}")

        with col2:
            st.markdown("### Rezultat Analiza Automata")
            if st.button("Porneste Scanarea AI"):
                with st.spinner('Se analizeaza structura tisulara...'):
                    start_time = time.time()
                    img_array = np.array(image)
                    results = model.predict(source=img_array, conf=conf_threshold)
                    end_time = time.time()
                    
                    res_plotted = results[0].plot(labels=True, boxes=True)
                    st.image(res_plotted, use_container_width=True, caption="Rezultat Detectie")
                    
                    inf_time = (end_time - start_time) * 1000

                    st.markdown(f"**Raport pentru:** {pacient_id} | {pacient_varsta} ani | Sex {pacient_sex}")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Timp Procesare", f"{inf_time:.2f} ms")
                    
                    num_boxes = len(results[0].boxes)
                    if num_boxes > 0:
                        conf_score = results[0].boxes.conf[0].item() * 100
                        m2.metric("Incredere Maxima", f"{conf_score:.1f}%")
                        st.error(f"ATENTIE: S-au detectat {num_boxes} zone cu potential patologic.")
                       
                        res_pil = Image.fromarray(res_plotted)
                        buf = io.BytesIO()
                        res_pil.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="💾 Descarca Rezultat Diagnostic",
                            data=byte_im,
                            file_name=f"Rezultat_{pacient_id}.png",
                            mime="image/png"
                        )
                    else:
                        m2.metric("Incredere", "N/A")
                        st.success("Analiza finalizata: Nu s-au detectat anomalii vizibile.")

with tab2:
    st.header("Performanta Modelului")
    st.write("Incarcati graficele de performanta obtinute in urma antrenarii pe GPU.")
    
    uploaded_charts = st.file_uploader(
        "Selectati: results.png, confusion_matrix.png", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        key="charts_uploader"
    )
    
    if uploaded_charts:
        col_a, col_b = st.columns(2)
        for file in uploaded_charts:
            name_lower = file.name.lower()
            img = Image.open(file)
            if "matrix" in name_lower:
                with col_b:
                    st.subheader("Matricea de Confuzie")
                    st.image(img, use_container_width=True)
            else:
                with col_a:
                    st.subheader("Evolutia Antrenarii")
                    st.image(img, use_container_width=True)
    else:
        st.info("Inca nu a fost incarcat niciun grafic.")

with tab3:
    st.header("📖 Ghid de Utilizare al Aplicatiei")
    st.markdown("""
    1. **Introducerea datelor:** Completati datele pacientului in panoul din stanga (ID, Varsta, Sex).
    2. **Incarcarea imaginii:** Trageti poza CT (slice axial) in zona centrala.
    3. **Setarea pragului:** Ajustati slider-ul de *Confidence*. Un prag de **0.45** este recomandat pentru a echilibra precizia cu rata de detectie.
    4. **Analiza:** Apasati butonul 'Porneste Scanarea AI'. Dacă se descopera o formatiune tumorala, veti avea optiunea de a salva rezultatul pe disc.
    """)